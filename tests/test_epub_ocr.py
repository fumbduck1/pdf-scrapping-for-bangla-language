"""Test script to verify OCR integration with EPUB scraper"""
import os
import tempfile
from pathlib import Path
import sys
from PIL import Image, ImageDraw, ImageFont
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_manager import create_job_config
from epub_scraper import run_epub_job
from deps import EPUBLIB_AVAILABLE


def test_epub_ocr_integration():
    """Test if OCR integration is working with EPUB scraper"""
    if not EPUBLIB_AVAILABLE:
        print("EPUB support not available - skipping OCR integration test")
        return
        
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        from ebooklib import epub
        book = epub.EpubBook()
        
        book.set_identifier('123456789')
        book.set_title('Test EPUB OCR Book')
        book.set_language('en')
        book.add_author('Test Author')
        
        # Create a chapter with an image
        c1 = epub.EpubHtml(title='Chapter 1', file_name='chap_01.xhtml', lang='en')
        c1.content = '''<html><head></head><body>
<h1>Chapter 1</h1>
<p>This chapter contains an image with text to OCR:</p>
<img src="image1.png" alt="Image with text" />
</body></html>'''
        
        book.add_item(c1)
        
        # Create a simple test image with text
        img_path = temp_dir / "image1.png"
        img = Image.new('RGB', (300, 150), color='white')
        draw = ImageDraw.Draw(img)
        
        # Use a simple font with larger size for better OCR
        try:
            # Try to use a system font
            draw.text((10, 20), "Hello World!", fill='black', font=ImageFont.truetype("arial.ttf", 30))
            draw.text((10, 60), "EPUB OCR Test", fill='black', font=ImageFont.truetype("arial.ttf", 24))
        except (OSError, IOError):
            # Fallback to default font if Arial not available
            draw.text((10, 20), "Hello World!", fill='black')
            draw.text((10, 60), "EPUB OCR Test", fill='black')
        
        img.save(img_path)
        
        # Add image to EPUB
        with open(img_path, 'rb') as f:
            img_content = f.read()
            
        i1 = epub.EpubImage(uid="image1", file_name="image1.png", media_type="image/png", content=img_content)
        book.add_item(i1)
        
        book.spine = ['nav', c1]
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        book.toc = [c1]
        
        test_epub_path = temp_dir / "test_ocr.epub"
        epub.write_epub(str(test_epub_path), book)
        
        print(f"Test EPUB file created: {test_epub_path}")
        
        job_config = create_job_config(
            input_path=str(test_epub_path),
            output_root=str(temp_dir),
            use_ocr=True,
            ocr_method='easyocr',
            ocr_lang='en',
            quality_mode=True,
            fast_mode=False,
            persist_renders=False,
        )
        
        print("Running EPUB scraping job with OCR...")
        result = run_epub_job(job_config, stop_event=None, log_cb=print)
        
        print(f"\nJob result:")
        print(f"  Scrape OK: {result.get('scrape_ok')}")
        print(f"  Save OK: {result.get('save_ok')}")
        print(f"  Output directory: {result.get('output_dir')}")
        
        if result.get('save_ok') and result.get('output_dir'):
            output_dir = Path(result.get('output_dir'))
            print(f"\nOutput files in {output_dir}:")
            for file in output_dir.glob('*'):
                print(f"  {file.name}")
                
            content_path = output_dir / "content.txt"
            if content_path.exists():
                print(f"\nContent extracted:")
                with open(content_path, 'r', encoding='utf-8') as f:
                    print(f.read())


if __name__ == "__main__":
    print("Testing EPUB scraper OCR integration")
    print("=" * 60)
    
    try:
        test_epub_ocr_integration()
        print("\n✓ EPUB OCR integration test completed")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        print(traceback.format_exc())
