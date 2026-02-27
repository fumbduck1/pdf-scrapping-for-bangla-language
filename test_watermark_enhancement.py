#!/usr/bin/env python3
"""
Test script for the 3-layer watermark enhancement system.
Processes a PDF with English watermarks and Bengali text.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scraper_old import PDFScraper
from config_manager import JobConfig

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def test_watermark_enhancement():
    """Test the watermark enhancement system"""

    pdf_path = r"C:\Users\akram\Desktop\pdf-scrapping-for-bangla-language\tests\data\sample.pdf"
    output_root = r"C:\Users\akram\Desktop\pdf-scrapping-for-bangla-language\test_output"

    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        return

    print_header("3-Layer Watermark Enhancement Test")

    print(f"📄 Input PDF: {Path(pdf_path).name}")
    print(f"📏 File size: {os.path.getsize(pdf_path) / (1024*1024):.2f} MB")
    print(f"🎯 Language: Bengali (ben)")
    print(f"⚙️ Mode: Quality (with watermark enhancement)")
    print(f"\n⏳ Processing started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Create output directory
        os.makedirs(output_root, exist_ok=True)

        # Initialize PDFScraper with watermark enhancement enabled
        scraper = PDFScraper(
            pdf_path=pdf_path,
            output_dir=os.path.join(output_root, "bengali_extraction"),
            ocr_lang="ben",
            quality_mode=True,  # Enables watermark detection
            fast_mode=False,
            persist_renders=False,
        )

        print_header("Processing Pages")

        # Process all pages
        success = scraper.scrape_all_pages()

        if not success:
            print("❌ PDF processing failed")
            return

        print("✅ Processing completed successfully\n")

        # Display results
        print_header("Extraction Results")

        results = scraper.results
        stats = results.get('statistics', {})
        pages = results.get('pages', {})

        print(f"📊 Statistics:")
        print(f"  • Total pages: {stats.get('total_pages', 'N/A')}")
        print(f"  • Pages with OCR: {stats.get('pages_with_ocr_text', 'N/A')}")
        print(f"  • Total text characters: {stats.get('total_text_length', 'N/A')}")
        print(f"  • OCR engine: {stats.get('ocr_method', 'N/A')}")

        # Show watermark tracking stats
        if hasattr(scraper, '_watermark_text_history'):
            print_header("Watermark Detection Results")

            watermark_history = scraper._watermark_text_history
            total_tracked = len(watermark_history)
            repeated_texts = {k: v for k, v in watermark_history.items() if len(v['pages']) >= 2}

            print(f"📍 Text fragments tracked: {total_tracked}")
            print(f"🔄 Fragments appearing on multiple pages (watermark candidates): {len(repeated_texts)}")

            if repeated_texts:
                print(f"\n🎯 High-confidence watermark texts:")
                for text_key, data in list(repeated_texts.items())[:5]:
                    avg_likelihood = sum(data['likelihoods']) / len(data['likelihoods']) if data['likelihoods'] else 0
                    print(f"   • '{text_key[:40]}...' - Pages: {sorted(data['pages'])}, Avg Likelihood: {avg_likelihood:.2f}")

        # Show per-page results
        print_header("Per-Page Extraction Summary")

        for page_key in sorted(pages.keys()):
            page_data = pages[page_key]
            page_num = page_key.replace('page_', '')
            content = page_data.get('content', '')
            confidence = page_data.get('ocr_page_confidence', 0)
            fragments = page_data.get('ocr_page_fragments', 0)

            char_count = len(content.strip())
            confidence_str = f"{confidence:.3f}" if confidence else "N/A"

            print(f"Page {page_num}: {char_count:5d} chars | Conf: {confidence_str} | Fragments: {fragments:3d}")

            # Show preview of extracted text
            if char_count > 0:
                preview = content.replace('\n', ' ')[:80]
                print(f"   Preview: {preview}...\n")

        # Save detailed report
        report_path = os.path.join(output_root, "watermark_test_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("3-LAYER WATERMARK ENHANCEMENT TEST REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Test Date: {datetime.now()}\n")
            f.write(f"PDF: {pdf_path}\n")
            f.write(f"Language: Bengali (ben)\n\n")

            f.write("STATISTICS\n")
            f.write("-"*70 + "\n")
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")

            f.write("\n\nWATERMARK DETECTION\n")
            f.write("-"*70 + "\n")
            if hasattr(scraper, '_watermark_text_history'):
                f.write(f"Total text fragments tracked: {len(scraper._watermark_text_history)}\n")
                repeated = {k: v for k, v in scraper._watermark_text_history.items() if len(v['pages']) >= 2}
                f.write(f"Watermark candidates (multi-page): {len(repeated)}\n\n")

                if repeated:
                    f.write("Detected Watermark Patterns:\n")
                    for text_key, data in repeated.items():
                        f.write(f"  • Text: {text_key}\n")
                        f.write(f"    Pages: {sorted(data['pages'])}\n")
                        f.write(f"    Appearances: {data['count']}\n")
                        avg_likelihood = sum(data['likelihoods']) / len(data['likelihoods']) if data['likelihoods'] else 0
                        f.write(f"    Avg Watermark Likelihood: {avg_likelihood:.3f}\n\n")

            f.write("\n\nPER-PAGE RESULTS\n")
            f.write("-"*70 + "\n")
            for page_key in sorted(pages.keys()):
                page_data = pages[page_key]
                page_num = page_key.replace('page_', '')
                f.write(f"\nPage {page_num}:\n")
                f.write(f"  Content Length: {len(page_data.get('content', ''))} chars\n")
                f.write(f"  Confidence: {page_data.get('ocr_page_confidence', 0):.3f}\n")
                f.write(f"  Fragments: {page_data.get('ocr_page_fragments', 0)}\n")

        print_header("Test Complete")
        print(f"✅ Report saved to: {report_path}")
        print(f"\n📁 Output directory: {os.path.join(output_root, 'bengali_extraction')}")

    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_watermark_enhancement()
