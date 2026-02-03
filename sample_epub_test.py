"""Create a sample EPUB file and test the scraper."""
import os
import tempfile
from pathlib import Path
from ebooklib import epub


def create_sample_epub(output_path: str):
    """Create a sample EPUB file for testing."""
    book = epub.EpubBook()
    
    book.set_identifier('123456789')
    book.set_title('Sample EPUB Book')
    book.set_language('en')
    book.add_author('John Doe')
    book.add_author('Jane Smith', file_as='Smith, Jane', role='ill')
    
    # Create chapters
    c1 = epub.EpubHtml(title='Chapter 1', file_name='chap_01.xhtml', lang='en')
    c1.content = '''<html><head></head><body>
<h1>Chapter 1: Introduction</h1>
<p>This is a sample EPUB book created for testing purposes.</p>
<p>It contains multiple chapters with different content types.</p>
</body></html>'''
    
    c2 = epub.EpubHtml(title='Chapter 2', file_name='chap_02.xhtml', lang='en')
    c2.content = '''<html><head></head><body>
<h1>Chapter 2: Getting Started</h1>
<p>In this chapter, we will learn about the basics of EPUB scraping.</p>
<ul>
    <li>What is an EPUB file?</li>
    <li>How does the scraper work?</li>
    <li>What can we extract?</li>
</ul>
</body></html>'''
    
    c3 = epub.EpubHtml(title='Chapter 3: Advanced Topics', file_name='chap_03.xhtml', lang='en')
    c3.content = '''<html><head></head><body>
<h1>Chapter 3: Advanced Topics</h1>
<p>This chapter covers more advanced topics in EPUB scraping.</p>
<p>Some of the topics include:</p>
<ol>
    <li>Handling different encoding formats</li>
    <li>Extracting metadata</li>
    <li>Processing images</li>
</ol>
</body></html>'''
    
    # Add chapters to book
    book.add_item(c1)
    book.add_item(c2)
    book.add_item(c3)
    
    # Create spine
    book.spine = ['nav', c1, c2, c3]
    
    # Add default NCX and Nav
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    
    # Add CSS
    style = '''
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 20px;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 5px;
        }
        p {
            margin-bottom: 15px;
        }
        ul, ol {
            margin-left: 25px;
            margin-bottom: 15px;
        }
        li {
            margin-bottom: 5px;
        }
    '''
    
    nav_css = epub.EpubItem(
        uid="style_nav",
        file_name="style/nav.css",
        media_type="text/css",
        content=style
    )
    book.add_item(nav_css)
    
    book.toc = (c1, c2, c3)
    
    epub.write_epub(output_path, book)
    print(f"Sample EPUB created at: {output_path}")


def test_scraper():
    """Test the EPUB scraper on the sample book."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        epub_path = temp_dir / "sample_book.epub"
        create_sample_epub(str(epub_path))
        
        from config_manager import create_job_config
        from epub_scraper import run_epub_job
        
        job_config = create_job_config(
            pdf_path=str(epub_path),
            output_root=str(temp_dir),
            use_ocr=False,
            ocr_method='easyocr',
            ocr_lang='en',
            quality_mode=True,
            fast_mode=False,
            persist_renders=False,
        )
        
        print("\nRunning EPUB scraping job...")
        result = run_epub_job(job_config, stop_event=None, log_cb=print)
        
        print(f"\nJob result:")
        print(f"  Scrape OK: {result.get('scrape_ok')}")
        print(f"  Save OK: {result.get('save_ok')}")
        print(f"  Output directory: {result.get('output_dir')}")
        
        if result.get('save_ok'):
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
    print("Testing EPUB scraper with sample book")
    print("=" * 50)
    test_scraper()
