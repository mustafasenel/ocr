import cv2
import fitz
from PIL import Image
import easyocr
import numpy as np
import pandas as pd
import os
import re

# Global DataFrame oluştur
global_df = pd.DataFrame(columns=["pdf_path", "sayi", "sirket_adi"])

def process_pdf(pdf_path):
    global global_df
    
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
            text_ = reader.readtext(img_original, min_size=50, detail=0, paragraph=True, width_ths=(0.3))

            # Şirket adını bul
            sirket_adi_regex = r'(\b\d{6}\b)\s*([\w\s\.\,ÇĞİÖŞÜ]{15,})(?=(?:Adres|$))'

            sirket_adi = [re.search(sirket_adi_regex, text) for text in text_]

            # Eğer eşleşen veri bulunduysa, ekrana yazdır ve data listesine ekle
            for t in sirket_adi:
                if t:
                    data.append({"pdf_path": pdf_path, "sayi": t.group(1), "sirket_adi": t.group(2)})

            # Eğer şirket adı bulunmuşsa, döngüyü sonlandır
            if data:
                found = True
            else:
                print("Şirket Adı Bulunamadı. Yeniden deneme...")
                attempts += 1

                # Eğer şirket adı bulunamadıysa resmi çevir ve tekrar dene
                img_original = cv2.rotate(img_original, cv2.ROTATE_90_CLOCKWISE)

        except Exception as e:
            print(f"Görüntü algılanamadı. Hata: {e}")
            data.append({"pdf_path": pdf_path, "sayi": "-", "sirket_adi": "-"})
            attempts += 1

    # Elde Edilen Verileri DataFrame'e Ekle
    global_df = pd.concat([global_df, pd.DataFrame(data)])

    # Her adımda elde edilen verileri CSV dosyasına yaz
    global_df.to_csv("elde_edilen_veriler.csv", index=False)

# Klasördeki tüm PDF dosyalarını işle
folder_path = "data"
for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(folder_path, filename)
        process_pdf(pdf_path)

# Elde edilen global DataFrame'i yazdır
print("Elde Edilen Veriler:")
print(global_df.to_string())
