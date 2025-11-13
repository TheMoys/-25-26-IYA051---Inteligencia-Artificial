import argparse
import os
import cv2
from utils import open_video_source, process_frame


OUTPUT_DIR = 'captures'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description='Reconocimiento de cartas - esqueleto')
    p.add_argument('--source', type=str, default='webcam',
    help='"webcam" | URL MJPEG/RTSP | path a imagen')
    return p.parse_args()

def main():
    args = parse_args()
    cap, is_image = open_video_source(args.source)


    frame_idx = 0
    if is_image:
        # Sólo procesar la imagen y salir
        ret, frame = cap.read()
        if not ret:
            print('No se pudo leer la imagen.')
            return
        out_frame = process_frame(frame)
        cv2.imshow('Salida', out_frame)
        print('Presiona cualquier tecla para cerrar...')
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return


    print('Presiona q para salir, p para guardar frame')
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Frame no disponible — intentando reconectar o terminando...')
            break


        out_frame = process_frame(frame)
        cv2.imshow('Video', out_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            fname = os.path.join(OUTPUT_DIR, f'capture_{frame_idx:04d}.jpg')
            cv2.imwrite(fname, frame)
            print('Guardado', fname)
            frame_idx += 1


    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()