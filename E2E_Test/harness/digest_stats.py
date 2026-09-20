"""Summarise digest quality signals for every E2E document (API side-channel)."""
import json, sys, urllib.request
BASE = "http://localhost:8022/api/v2"
tok = json.load(urllib.request.urlopen(urllib.request.Request(f"{BASE}/auth/login", data=b'{"username":"admin","password":"admin"}', headers={"Content-Type": "application/json"})))["access_token"]
def get(p, method="GET"):
    r = urllib.request.Request(f"{BASE}/{p}", method=method, headers={"Authorization": f"Bearer {tok}"})
    return json.load(urllib.request.urlopen(r))
for doc in sys.argv[1:]:
    d = get(f"documents/{doc}/digest", "POST")
    ch = (d.get("main_content") or {}).get("chapters") or []
    lens = [len(c.get("content") or "") for c in ch]
    print(f"{doc} {d['original_filename'][:40]:40} lang={d.get('source_language')} chapters={len(ch)} "
          f"chars(min/avg/max)={min(lens or [0])}/{sum(lens)//max(1,len(lens))}/{max(lens or [0])} "
          f"kw={len(d.get('keywords') or [])} rd={len(d.get('research_directions') or [])} "
          f"abstract={len(d.get('abstract') or '')} missing={d.get('missing')}")
    for c in ch[:12]:
        print(f"     {c.get('number')}. {str(c.get('title_vi'))[:60]} | {str(c.get('title_original'))[:40]} | {len(c.get('content') or '')}c")
