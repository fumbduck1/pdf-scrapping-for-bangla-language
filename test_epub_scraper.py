"""Test script for EPUB scraper functionality"""
import os
import tempfile
from pathlib import Path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_manager import create_job_config
from epub_scraper import run_epub_job, EPUBScraper
from deps import EPUBLIB_AVAILABLE


def test_epub_support_available():
    """Test if EPUB support is available"""
    print(f"EPUB support available: {EPUBLIB_AVAILABLE}")
    assert EPUBLIB_AVAILABLE, "EPUB support not available"


def test_run_epub_job():
    """Test running an EPUB job"""
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Create a simple test EPUB file
        test_epub_path = temp_dir / "test.epub"
        
        # Create minimal EPUB content
        from ebooklib import epub
        book = epub.EpubBook()
        
        # Set metadata
        book.set_identifier('123456789')
        book.set_title('Test EPUB Book')
        book.set_language('en')
        book.add_author('Test Author')
        
        # Create a chapter
        c1 = epub.EpubHtml(title='Chapter 1', file_name='chap_01.xhtml', lang='en')
        c1.content = '''<html><head></head><body>
<h1>Chapter 1</h1>
<p>This is a test chapter with some text.</p>
<p>More text here.</p>
</body></html>'''
        
        # Add chapter
        book.add_item(c1)
        
        # Create spine
        book.spine = ['nav', c1]
        
        # Add default NCX and Nav
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        
        # Add chapter to book
        book.toc = [c1]
        
        # Write the EPUB file
        epub.write_epub(str(test_epub_path), book)
        
        print(f"Test EPUB file created: {test_epub_path}")
        
        # Create job configuration
        job_config = create_job_config(
            pdf_path=str(test_epub_path),
            output_root=str(temp_dir),
            use_ocr=True,
            ocr_method='easyocr',
            ocr_lang='en',
            quality_mode=True,
            fast_mode=False,
            persist_renders=False,
        )
        
        print("Running EPUB scraping job...")
        result = run_epub_job(job_config, stop_event=None, log_cb=print)
        
        print(f"\nJob result:")
        print(f"  Scrape OK: {result.get('scrape_ok')}")
        print(f"  Save OK: {result.get('save_ok')}")
        print(f"  Output directory: {result.get('output_dir')}")
        
        if result.get('save_ok'):
            # Check output files
            output_dir = Path(result.get('output_dir'))
            print(f"\nOutput files in {output_dir}:")
            for file in output_dir.glob('*'):
                print(f"  {file.name}")
                
            # Check content.txt
            content_path = output_dir / "content.txt"
            if content_path.exists():
                print(f"\nContent extracted:")
                with open(content_path, 'r', encoding='utf-8') as f:
                    print(f.read())


def test_text_structure_preservation():
    """Test that text structure is preserved with paragraphs and line breaks."""
    print("\nTesting text structure preservation...")
    
    # Create a simple test EPUB with structured content
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        test_epub_path = temp_dir / "test_structure.epub"
        
        from ebooklib import epub
        book = epub.EpubBook()
        
        book.set_identifier('123456789')
        book.set_title('Test Book with Structure')
        book.set_language('en')
        book.add_author('Test Author')
        
        # Create a chapter with proper HTML structure
        c1 = epub.EpubHtml(title='Chapter 1', file_name='chap_01.xhtml', lang='en')
        c1.content = '''<html><head></head><body>
<h1>Chapter 1: Introduction</h1>
<p>This is the first paragraph. It contains several sentences.</p>
<p>This is the second paragraph.</p>
<blockquote>
  <p>This is a blockquote paragraph.</p>
  <p>Another blockquote line.</p>
</blockquote>
<p>This is the third paragraph with some <strong>bold text</strong> and <em>italic text</em>.</p>
</body></html>'''
        
        book.add_item(c1)
        book.spine = ['nav', c1]
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        book.toc = [c1]
        
        epub.write_epub(str(test_epub_path), book)
        
        # Create job configuration
        job_config = create_job_config(
            pdf_path=str(test_epub_path),
            output_root=str(temp_dir),
            use_ocr=True,
            ocr_method='easyocr',
            ocr_lang='en',
            quality_mode=True,
            fast_mode=False,
            persist_renders=False,
        )
        
        result = run_epub_job(job_config, stop_event=None, log_cb=lambda x: None)
        
        assert result.get('scrape_ok'), "Scraping failed"
        assert result.get('save_ok'), "Saving failed"
        
        output_dir = Path(result.get('output_dir'))
        content_path = output_dir / "content.txt"
        
        assert content_path.exists(), "Content file not created"
        
        with open(content_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check that paragraphs are preserved
        assert 'Chapter 1: Introduction' in content, "Chapter title not found"
        assert 'This is the first paragraph' in content, "First paragraph not found"
        assert 'This is the second paragraph' in content, "Second paragraph not found"
        assert 'This is a blockquote paragraph' in content, "Blockquote paragraph not found"
        
        # Check that we have newlines (not all content in one line)
        paragraph_count = content.count('\n')
        assert paragraph_count > 5, f"Expected more than 5 paragraphs, got {paragraph_count}"
        
        print("✓ Text structure preserved with paragraphs and line breaks")


if __name__ == "__main__":
    print("Testing EPUB scraper functionality")
    print("=" * 50)
    
    try:
        test_epub_support_available()
        print("✓ EPUB support available")
        
        test_run_epub_job()
        test_text_structure_preservation()
        print("\n✓ All EPUB scraping tests completed successfully")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        print(traceback.format_exc())
