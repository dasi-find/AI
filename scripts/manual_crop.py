from pathlib import Path

import cv2
import numpy as np


# ==========================================
# 설정
# ==========================================

INPUT_DIR = Path("wallet_dataset")
OUTPUT_DIR = Path("wallet_dataset_crop")

PADDING_RATIO = 0.10

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# 한글 경로 대응 이미지 읽기
# ==========================================

def imread_unicode(path):
    data = np.fromfile(
        str(path),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        data,
        cv2.IMREAD_COLOR
    )

    return image


# ==========================================
# 한글 경로 대응 이미지 저장
# ==========================================

def imwrite_unicode(path, image):

    extension = path.suffix

    success, encoded = cv2.imencode(
        extension,
        image
    )

    if not success:
        return False

    encoded.tofile(
        str(path)
    )

    return True


# ==========================================
# Bounding Box에 Padding 추가
# ==========================================

def add_padding(
    x,
    y,
    w,
    h,
    image_width,
    image_height,
    padding_ratio=0.10
):

    padding_x = int(
        w * padding_ratio
    )

    padding_y = int(
        h * padding_ratio
    )

    x1 = max(
        0,
        x - padding_x
    )

    y1 = max(
        0,
        y - padding_y
    )

    x2 = min(
        image_width,
        x + w + padding_x
    )

    y2 = min(
        image_height,
        y + h + padding_y
    )

    return x1, y1, x2, y2


# ==========================================
# 데이터셋 가져오기
# ==========================================

image_files = sorted(
    list(
        INPUT_DIR.glob("*.jpg")
    )
)

print(
    f"총 이미지 수: {len(image_files)}"
)

print()
print("=====================================")
print("수동 Crop 방법")
print("=====================================")
print("1. 마우스로 지갑 전체를 드래그")
print("2. 지갑 테두리 기준으로 선택")
print("3. ENTER 또는 SPACE → 확정")
print("4. C → 선택 취소")
print()
print(
    f"선택 영역에 자동으로 "
    f"{PADDING_RATIO * 100:.0f}% 여백을 추가합니다."
)
print("=====================================")
print()


# ==========================================
# 이미지 순차 Crop
# ==========================================

for index, image_path in enumerate(
    image_files,
    start=1
):

    print()
    print(
        f"[{index}/{len(image_files)}] "
        f"{image_path.name}"
    )

    image = imread_unicode(
        image_path
    )

    if image is None:

        print(
            "❌ 이미지를 읽을 수 없습니다."
        )

        continue

    height, width = image.shape[:2]

    # 너무 큰 사진은 화면에 맞게 축소해서 보여줌
    max_display_width = 1200
    max_display_height = 800

    scale = min(
        max_display_width / width,
        max_display_height / height,
        1.0
    )

    display_width = int(
        width * scale
    )

    display_height = int(
        height * scale
    )

    display_image = cv2.resize(
        image,
        (
            display_width,
            display_height
        )
    )

    window_name = (
        f"Crop: {image_path.name}"
    )

    roi = cv2.selectROI(
        window_name,
        display_image,
        showCrosshair=True,
        fromCenter=False
    )

    cv2.destroyWindow(
        window_name
    )

    x, y, w, h = roi

    # 선택 취소
    if w == 0 or h == 0:

        print(
            "⚠️ Crop 선택이 취소되었습니다."
        )

        continue

    # 화면 축소 비율을 원본 좌표로 변환
    x = int(x / scale)
    y = int(y / scale)
    w = int(w / scale)
    h = int(h / scale)

    # 10% padding 추가
    x1, y1, x2, y2 = add_padding(
        x,
        y,
        w,
        h,
        width,
        height,
        PADDING_RATIO
    )

    cropped = image[
        y1:y2,
        x1:x2
    ]

    output_path = (
        OUTPUT_DIR
        / f"crop_{image_path.name}"
    )

    success = imwrite_unicode(
        output_path,
        cropped
    )

    if success:

        print(
            "✅ 저장:",
            output_path
        )

        print(
            f"원본 크기: "
            f"{width} x {height}"
        )

        print(
            f"Crop 크기: "
            f"{x2 - x1} x {y2 - y1}"
        )

    else:

        print(
            "❌ 저장 실패:",
            output_path
        )


cv2.destroyAllWindows()

print()
print("=====================================")
print("Crop 작업 완료")
print("=====================================")