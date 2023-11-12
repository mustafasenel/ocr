import cv2
import fitz
from PIL import Image
import easyocr
import numpy as np
import matplotlib.pyplot as plt
import re
import os

def process_pdf(pdf_path):
    # PDF dosyasını aç
    pdf_document = fitz.open(pdf_path)
    # Sayfayı al
    page = pdf_document[0]

    # Görüntüyü PIL Image formatına dönüştür
    image = page.get_pixmap(dpi=700)
    pil_image = Image.frombytes("RGB", [image.width, image.height], image.samples)

    # PIL Image'i NumPy array'e dönüştür
    img_array = np.array(pil_image)

    img_array_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_original = cv2.rotate(img_array_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # OCR için dil ve GPU ayarları
    reader = easyocr.Reader(['tr'], gpu=False)

    data = []
    max_attempts = 4  # Maksimum deneme sayısı

    attempts = 0
    found = False
    while not found and attempts < max_attempts:
        try:
            # OCR işlemini gerçekleştir
            text_ = reader.readtext(img_original, min_size=50, detail=1, paragraph=True, width_ths=(0.3))
            # # Sadece metin kısmını al
            # text_only = [''.join(box[1]) if isinstance(box[1], list) else box[1] for box_list in text_ for box in box_list]
            # print(text_only)
            # # Şirket adını bul
            # sirket_adi_regex = r'(\b00\d{4}\b)(.*?)(?=(?:Adres|$))'
            # sirket_adi = [re.search(sirket_adi_regex, text) for text in text_only]

            # Eğer eşleşen veri bulunduysa, ekrana yazdır ve data listesine ekle
            # for t in sirket_adi:
            #     if t:
                    # data.append({"sayı": t.group(1), "Şirket Adı": t.group(2)})

            # Eğer şirket adı bulunmuşsa, döngüyü sonlandır
            if True:
                found = True
                try:
                    for box_info in text_:
                        # Bounding box koordinatlarını al
                        box_coordinates = np.array(box_info[0])
                        # Koordinatları tam sayıya dönüştür
                        box_coordinates = box_coordinates.astype(int)

                        # Bounding box'ı çiz
                        img_original = cv2.polylines(img_original, [box_coordinates], isClosed=True, color=(0, 255, 0), thickness=2)

                        # Metni ekle (isteğe bağlı)
                        text = box_info[1]
                        cv2.putText(img_original, text, (box_coordinates[0][0], box_coordinates[0][1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                except:
                    print(f"Error processing {pdf_path}")

                print(f"Text from {pdf_path}:")
                plt.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
                plt.show()
            else:
                print("Şirket Adı Bulunamadı. Yeniden deneme...")
                attempts += 1

                plt.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
                plt.show()
                # Eğer şirket adı bulunamadıysa resmi çevir ve tekrar dene
                img_original = cv2.rotate(img_original, cv2.ROTATE_90_CLOCKWISE)

        except Exception as e:
            print(f"Görüntü algılanamadı. Hata: {e}")
            attempts += 1

    # Elde Edilen Verileri Ekrana Yazdır
    print("Elde Edilen Veriler:")
    for item in data:
        print(item)


# Klasördeki tüm PDF dosyalarını işle
folder_path = "data" 
for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(folder_path, filename)
        process_pdf(pdf_path)
