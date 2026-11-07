"""
fits_utils.py — Модуль чтения и записи FITS-файлов.
Использует astropy.io.fits при наличии, либо встроенный совместимый парсер стандарта FITS (RFC 4047).
"""
import os
import numpy as np

try:
    from astropy.io import fits as _astropy_fits
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False


def read_fits(filepath):
    """
    Читает FITS-файл и возвращает (data, header_dict).
    Учитывает BSCALE и BZERO при их наличии.
    """
    if HAS_ASTROPY:
        with _astropy_fits.open(filepath) as hdul:
            hdu = hdul[0]
            data = hdu.data.copy()
            hdr = dict(hdu.header)
            return data, hdr
    else:
        with open(filepath, 'rb') as f:
            header = {}
            while True:
                block = f.read(2880)
                if not block:
                    break
                ended = False
                for i in range(0, 2880, 80):
                    card = block[i:i+80].decode('ascii', errors='replace')
                    key = card[:8].strip()
                    if key == 'END':
                        ended = True
                        break
                    if '=' in card[:10]:
                        val_comment = card[10:].split('/')
                        val_str = val_comment[0].strip()
                        if val_str.startswith("'") and val_str.endswith("'"):
                            val = val_str[1:-1].strip()
                        elif val_str in ('T', 'F'):
                            val = (val_str == 'T')
                        else:
                            try:
                                if '.' in val_str or 'E' in val_str or 'e' in val_str:
                                    val = float(val_str)
                                else:
                                    val = int(val_str)
                            except ValueError:
                                val = val_str
                        header[key] = val
                if ended:
                    break

            bitpix = int(header.get('BITPIX', -32))
            naxis1 = int(header.get('NAXIS1', 1024))
            naxis2 = int(header.get('NAXIS2', 1024))
            dtype_map = {-32: '>f4', -64: '>f8', 16: '>i2', 32: '>i4', 8: '>u1'}
            dt = dtype_map.get(bitpix, '>f4')
            nbytes = naxis1 * naxis2 * (abs(bitpix) // 8)
            raw_bytes = f.read(nbytes)
            data = np.frombuffer(raw_bytes, dtype=dt).reshape((naxis2, naxis1)).copy()
            bscale = float(header.get('BSCALE', 1.0))
            bzero = float(header.get('BZERO', 0.0))
            if bscale != 1.0 or bzero != 0.0:
                data = data.astype(np.float64) * bscale + bzero
            return data, header


def write_fits(filepath, data, header_dict=None, preserve_dtype=np.float32):
    """
    Записывает 2D массив в стандартный FITS-файл с корректным заголовком и выравниванием по блокам 2880 байт.
    Автоматически создаёт родительскую директорию, если она не существует.
    """
    parent_dir = os.path.dirname(filepath)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    if preserve_dtype is not None:
        out_data = data.astype(preserve_dtype)
    else:
        out_data = data.copy()

    if HAS_ASTROPY:
        hdu = _astropy_fits.PrimaryHDU(out_data)
        if header_dict:
            for k, v in header_dict.items():
                if k not in ('SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'EXTEND', 'COMMENT', 'HISTORY', ''):
                    try:
                        hdu.header[k[:8]] = v
                    except Exception:
                        pass
        hdu.writeto(filepath, overwrite=True)
    else:
        dtype_to_bitpix = {
            np.dtype('float32'): (-32, '>f4'),
            np.dtype('>f4'): (-32, '>f4'),
            np.dtype('float64'): (-64, '>f8'),
            np.dtype('>f8'): (-64, '>f8'),
            np.dtype('int16'): (16, '>i2'),
            np.dtype('>i2'): (16, '>i2'),
            np.dtype('int32'): (32, '>i4'),
            np.dtype('>i4'): (32, '>i4'),
            np.dtype('uint8'): (8, '>u1'),
        }
        bitpix, big_dt = dtype_to_bitpix.get(out_data.dtype, (-32, '>f4'))
        naxis2, naxis1 = out_data.shape

        cards = []
        def add_card(k, v, comment=''):
            k_str = f"{k[:8]:<8s}= "
            if isinstance(v, bool):
                v_str = f"{'T' if v else 'F':>20s}"
            elif isinstance(v, int):
                v_str = f"{v:>20d}"
            elif isinstance(v, float):
                v_str = f"{v:>20.8G}"
            else:
                s = str(v)[:18]
                v_str = f"'{s:<18s}'"
            line = f"{k_str}{v_str}"
            if comment:
                line += f" / {comment}"
            cards.append(f"{line:<80s}"[:80])

        add_card('SIMPLE', True, 'conforms to FITS standard')
        add_card('BITPIX', bitpix, 'array data type')
        add_card('NAXIS', 2, 'number of array dimensions')
        add_card('NAXIS1', naxis1, 'width')
        add_card('NAXIS2', naxis2, 'height')

        if header_dict:
            skip = {'SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'END', 'COMMENT', 'HISTORY', ''}
            for k, v in header_dict.items():
                if k not in skip and isinstance(v, (bool, int, float, str)):
                    add_card(k, v)

        cards.append(f"{'END':<80s}")
        header_bytes = "".join(cards).encode('ascii', errors='replace')
        pad_hdr = (2880 - (len(header_bytes) % 2880)) % 2880
        header_bytes += b' ' * pad_hdr

        data_bytes = out_data.astype(big_dt).tobytes()
        pad_data = (2880 - (len(data_bytes) % 2880)) % 2880
        data_bytes += b'\x00' * pad_data

        with open(filepath, 'wb') as f:
            f.write(header_bytes)
            f.write(data_bytes)