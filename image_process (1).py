import tensorflow as tf
import numpy as np
from urllib.request import urlopen
model=tf.keras.models.load_model('new_lenet5')

formats = list(map(bytes, ['BMP', 'PNG', 'RAW'], ['utf-8']*3)) + [b'\xff\xd8']
decoders = {
    b'BMP': tf.io.decode_bmp,
    b'PNG': tf.io.decode_png,
    b'RAW': tf.io.decode_raw,
    b'\xff\xd8': tf.io.decode_jpeg
}

def process_img(url):
    
        img=urlopen(url)
        img=img.read()
        for format in formats:
            if format in img:
                print("Format Found")
                img = decoders[format](img, channels = 1)                
                print("Image decoded")
                image=tf.io.encode_jpeg(img,format='grayscale')
                print("Image encoded")
                tf.io.write_file('./static/image.jpeg',image)
                img = tf.image.resize(img, [28, 28])
                img = tf.expand_dims(img,0)
                print(img.shape)
                res=model(img)
                return tf.argmax(res, 1).numpy()[0]

if __name__=='__main__':
    link="data:image/png;base64,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"
    print(process_img(link))
    # print(bytes('JFIF','utf-8') in process_img(link))
    
    