"""
Quick verification script to test thinning fix.
Run directly to verify the bug is fixed.
"""
import sys
sys.path.insert(0, 'e:\\AI_Project\\OCR_Quantization\\docuflow')

from spatial.thinning import hierarchical_thinning


def test_barrier_prevents_merge():
    """Test that barriers prevent merging."""
    print("\n=== Test 1: Barrier prevents merge ===")
    nodes = [
        {
            'label': 'text',
            'page_number': 1,
            'bbox_y1': 0, 'bbox_y2': 10,
            'bbox_x1': 0, 'bbox_x2': 100,
            'text_content': 'Paragraph before equation'
        },
        {
            'label': 'equation',
            'page_number': 1,
            'bbox_y1': 15, 'bbox_y2': 30,
            'bbox_x1': 0, 'bbox_x2': 100,
            'text_content': 'E = mc^2'
        },
        {
            'label': 'text',
            'page_number': 1,
            'bbox_y1': 35, 'bbox_y2': 45,
            'bbox_x1': 0, 'bbox_x2': 100,
            'text_content': 'Paragraph after equation'
        }
    ]
    
    result = hierarchical_thinning(nodes, use_dynamic_gap=False, gap_threshold_multiplier=5.0)
    
    print(f"Input nodes: {len(nodes)}")
    print(f"Output nodes: {len(result)}")
    print(f"Labels: {[n['label'] for n in result]}")
    
    if len(result) == 3:
        print("[PASS] Barrier prevents merge correctly")
        return True
    else:
        print("[FAIL] Expected 3 nodes (text, equation, text)")
        return False


def test_merge_consecutive_text():
    """Test that consecutive text blocks merge."""
    print("\n=== Test 2: Merge consecutive text ===")
    nodes = [
        {
            'label': 'text',
            'page_number': 1,
            'bbox_y1': 0, 'bbox_y2': 10,
            'bbox_x1': 0, 'bbox_x2': 100,
            'text_content': 'Line 1'
        },
        {
            'label': 'text',
            'page_number': 1,
            'bbox_y1': 12, 'bbox_y2': 22,
            'bbox_x1': 0, 'bbox_x2': 100,
            'text_content': 'Line 2'
        },
        {
            'label': 'text',
            'page_number': 1,
            'bbox_y1': 24, 'bbox_y2': 34,
            'bbox_x1': 0, 'bbox_x2': 100,
            'text_content': 'Line 3'
        }
    ]
    
    result = hierarchical_thinning(nodes, use_dynamic_gap=False, gap_threshold_multiplier=5.0)
    
    print(f"Input nodes: {len(nodes)}")
    print(f"Output nodes: {len(result)}")
    if result:
        print(f"Label: {result[0]['label']}")
        print(f"Text content: {result[0]['text_content'][:50]}...")
        print(f"Merged from: {result[0].get('merged_from', 1)} nodes")
    
    if len(result) == 1 and result[0]['label'] == 'paragraph':
        print("[PASS] Text blocks merged correctly")
        return True
    else:
        print("[FAIL] Expected 1 merged paragraph")
        return False


def test_no_cross_page_merge():
    """Test that cross-page merging is prevented."""
    print("\n=== Test 3: No cross-page merge ===")
    nodes = [
        {
            'label': 'text',
            'page_number': 1,
            'bbox_y1': 500, 'bbox_y2': 510,
            'bbox_x1': 0, 'bbox_x2': 100,
            'text_content': 'End of page 1'
        },
        {
            'label': 'text',
            'page_number': 2,
            'bbox_y1': 0, 'bbox_y2': 10,
            'bbox_x1': 0, 'bbox_x2': 100,
            'text_content': 'Start of page 2'
        }
    ]
    
    result = hierarchical_thinning(nodes, use_dynamic_gap=False, gap_threshold_multiplier=100.0)
    
    print(f"Input nodes: {len(nodes)} (from 2 different pages)")
    print(f"Output nodes: {len(result)}")
    print(f"Pages: {[n.get('page_number') for n in result]}")
    
    if len(result) == 2:
        print("[PASS] No cross-page merge")
        return True
    else:
        print("[FAIL] Expected 2 nodes (no cross-page merge)")
        return False


def test_complex_scenario():
    """Test realistic scenario with mixed content."""
    print("\n=== Test 4: Complex scenario ===")
    nodes = [
        {'label': 'title', 'page_number': 1, 'bbox_y1': 0, 'bbox_y2': 20,
         'bbox_x1': 0, 'bbox_x2': 100, 'text_content': 'Introduction'},
        
        {'label': 'text', 'page_number': 1, 'bbox_y1': 25, 'bbox_y2': 35,
         'bbox_x1': 0, 'bbox_x2': 100, 'text_content': 'Para 1 line 1'},
        {'label': 'text', 'page_number': 1, 'bbox_y1': 37, 'bbox_y2': 47,
         'bbox_x1': 0, 'bbox_x2': 100, 'text_content': 'Para 1 line 2'},
        
        {'label': 'equation', 'page_number': 1, 'bbox_y1': 55, 'bbox_y2': 70,
         'bbox_x1': 0, 'bbox_x2': 100, 'text_content': 'a^2 + b^2 = c^2'},
        
        {'label': 'text', 'page_number': 1, 'bbox_y1': 75, 'bbox_y2': 85,
         'bbox_x1': 0, 'bbox_x2': 100, 'text_content': 'Para 2 line 1'},
        {'label': 'text', 'page_number': 1, 'bbox_y1': 87, 'bbox_y2': 97,
         'bbox_x1': 0, 'bbox_x2': 100, 'text_content': 'Para 2 line 2'},
    ]
    
    result = hierarchical_thinning(nodes, use_dynamic_gap=False, gap_threshold_multiplier=5.0)
    
    print(f"Input nodes: {len(nodes)}")
    print(f"Output nodes: {len(result)}")
    print(f"Structure:")
    for i, n in enumerate(result):
        label = n['label']
        merged = n.get('merged_from', 1)
        print(f"  {i+1}. {label} (merged from {merged} nodes)")
    
    # Expected: title, paragraph(2), equation, paragraph(2) = 4 nodes
    if len(result) == 4:
        print("[PASS] Complex scenario handled correctly")
        return True
    else:
        print("[FAIL] Expected 4 nodes")
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("THINNING FIX VERIFICATION")
    print("=" * 60)
    
    results = []
    results.append(test_barrier_prevents_merge())
    results.append(test_merge_consecutive_text())
    results.append(test_no_cross_page_merge())
    results.append(test_complex_scenario())
    
    print("\n" + "=" * 60)
    print(f"SUMMARY: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)
    
    if all(results):
        print("\n[SUCCESS] ALL TESTS PASSED - Bug is fixed!")
        sys.exit(0)
    else:
        print("\n[ERROR] SOME TESTS FAILED - Bug may still exist")
        sys.exit(1)
