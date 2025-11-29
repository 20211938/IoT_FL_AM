"""
data 폴더 내의 모든 .tif 파일을 .jpg로 변환하는 스크립트
"""
import os
from pathlib import Path
from PIL import Image
import argparse

def convert_tif_to_jpg(data_dir, delete_original=True, quality=95):
    """
    data 폴더 내의 모든 .tif 파일을 .jpg로 변환합니다.
    
    Args:
        data_dir: 변환할 데이터 폴더 경로
        delete_original: 원본 .tif 파일 삭제 여부 (기본값: True)
        quality: JPG 품질 (1-100, 기본값: 95)
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"오류: {data_dir} 폴더를 찾을 수 없습니다.")
        return
    
    # 모든 .tif 파일 찾기
    tif_files = list(data_path.rglob("*.tif"))
    
    if not tif_files:
        print("변환할 .tif 파일이 없습니다.")
        return
    
    print(f"총 {len(tif_files)}개의 .tif 파일을 찾았습니다.")
    
    converted_count = 0
    error_count = 0
    
    for tif_file in tif_files:
        try:
            # .jpg 파일 경로 생성
            jpg_file = tif_file.with_suffix('.jpg')
            
            # 이미지 읽기
            with Image.open(tif_file) as img:
                # RGB 모드로 변환 (RGBA나 다른 모드인 경우)
                if img.mode in ('RGBA', 'LA', 'P'):
                    # 투명도가 있는 경우 흰색 배경으로 변환
                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                    img = rgb_img
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # .jpg로 저장
                img.save(jpg_file, 'JPEG', quality=quality, optimize=True)
            
            print(f"변환 완료: {tif_file.name} -> {jpg_file.name}")
            
            # 원본 파일 삭제 (옵션)
            if delete_original:
                tif_file.unlink()
                print(f"  원본 파일 삭제: {tif_file.name}")
            
            converted_count += 1
            
        except Exception as e:
            print(f"오류 발생 ({tif_file}): {str(e)}")
            error_count += 1
    
    print(f"\n변환 완료!")
    print(f"  성공: {converted_count}개")
    print(f"  실패: {error_count}개")
    if delete_original:
        print(f"  원본 .tif 파일이 삭제되었습니다.")
    else:
        print(f"  원본 .tif 파일이 유지되었습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='data 폴더 내의 .tif 파일을 .jpg로 변환')
    parser.add_argument('--data-dir', type=str, default='dataset',
                        help='변환할 데이터 폴더 경로 (기본값: data)')
    parser.add_argument('--delete-original', action='store_true', default=True,
                        help='원본 .tif 파일 삭제 (기본값: True)')
    parser.add_argument('--keep-original', action='store_false', dest='delete_original',
                        help='원본 .tif 파일 유지')
    parser.add_argument('--quality', type=int, default=95,
                        help='JPG 품질 (1-100, 기본값: 95)')
    
    args = parser.parse_args()
    
    # 사용자 확인 (원본 삭제 옵션이 켜진 경우)
    if args.delete_original:
        response = input("경고: 원본 .tif 파일이 삭제됩니다. 계속하시겠습니까? (yes/no): ")
        if response.lower() != 'yes':
            print("작업이 취소되었습니다.")
            exit(0)
    
    convert_tif_to_jpg(args.data_dir, args.delete_original, args.quality)

