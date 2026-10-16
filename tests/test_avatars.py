"""AvatarService: FPL photo codes without a photo on the CDN fall through to Wikipedia,
and peek() never hands out a photo URL that is known not to exist (mocked HTTP, no network)."""
import asyncio
import time

import httpx

from services.avatars import FPL_PHOTO_URL, AvatarService, _title_matches

CHERIF, HAALAND = 627221, 223094          # no photo on the CDN / has one
CHERIF_PHOTO = "https://upload.wikimedia.org/commons/thumb/Sidiki_Cherif.jpg"
BOOTSTRAP = {
    "teams": [{"id": 1, "name": "Coventry City"}, {"id": 2, "name": "Man City"}],
    "elements": [
        {"code": CHERIF, "first_name": "Sidiki", "second_name": "Cherif", "web_name": "Cherif", "team": 1},
        {"code": HAALAND, "first_name": "Erling", "second_name": "Haaland", "web_name": "Haaland", "team": 2},
    ],
}
# The search's top hit is the player's page; a disambiguation page comes back too
WIKI_SEARCH = {"query": {"pages": [
    {"title": "Sidiki", "index": 2, "thumbnail": {"source": "https://upload.wikimedia.org/x/Sidiki_village.jpg"}},
    {"title": "Sidiki Cherif", "index": 1, "thumbnail": {"source": CHERIF_PHOTO}},
]}}


def _service(tmp_path, cdn=None, wiki=WIKI_SEARCH, cache=None):
    """An AvatarService on a mock transport. `cdn` maps an FPL code to the CDN's HEAD status."""
    cdn = {CHERIF: 403, HAALAND: 200} if cdn is None else cdn
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, str(request.url)))
        if "bootstrap-static" in request.url.path:
            return httpx.Response(200, json=BOOTSTRAP)
        if request.url.host == "resources.premierleague.com":
            code = int(request.url.path.rsplit("/", 1)[-1].split(".")[0])
            status = cdn.get(code, 403)
            if isinstance(status, Exception):
                raise status
            return httpx.Response(status)
        if request.url.host == "en.wikipedia.org":
            return httpx.Response(200, json=wiki)
        return httpx.Response(404)

    svc = AvatarService(cache_path=str(tmp_path / "avatars.json"))
    svc.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    if cache:
        svc._cache.update(cache)
    return svc, calls


async def _loaded(svc):
    await svc._load_fpl_codes()
    await svc._check_fpl_photos([CHERIF, HAALAND])


def test_missing_fpl_photo_falls_back_to_wikipedia(tmp_path):
    async def run():
        svc, calls = _service(tmp_path)
        await _loaded(svc)
        assert svc.peek("Erling Haaland") == FPL_PHOTO_URL.format(code=HAALAND)
        assert svc.peek("Sidiki Cherif") is None              # no broken FPL link
        assert await svc.get("Sidiki Cherif", "Coventry City") == CHERIF_PHOTO
        assert svc.peek("Sidiki Cherif") == CHERIF_PHOTO      # cached for later pages
        # Each code is checked on the CDN once
        heads = [u for m, u in calls if m == "HEAD"]
        assert sorted(heads) == sorted(FPL_PHOTO_URL.format(code=c) for c in (CHERIF, HAALAND))
    asyncio.run(run())


def test_cached_fpl_url_without_a_photo_is_replaced(tmp_path):
    async def run():
        # Cached before the CDN checks existed: the FPL URL for a code with no photo
        stale = {"sidiki cherif": (FPL_PHOTO_URL.format(code=CHERIF), time.time())}
        svc, _ = _service(tmp_path, cache=stale)
        await _loaded(svc)
        assert svc.peek("Sidiki Cherif") is None
        assert await svc.get("Sidiki Cherif") == CHERIF_PHOTO
    asyncio.run(run())


def test_no_photo_anywhere_is_a_cached_miss(tmp_path):
    async def run():
        svc, calls = _service(tmp_path, wiki={"query": {"pages": []}})
        await _loaded(svc)
        assert await svc.get("Sidiki Cherif") is None
        n = len(calls)
        assert await svc.get("Sidiki Cherif") is None
        assert len(calls) == n                                # no repeat lookups
    asyncio.run(run())


def test_peek_with_a_graph_code(tmp_path):
    async def run():
        svc, _ = _service(tmp_path)
        await _loaded(svc)
        # Names the feed doesn't know (e.g. a different spelling in the graph)
        assert svc.peek("S. Cherif", fpl_code=CHERIF) is None
        assert svc.peek("E. Haaland", fpl_code=HAALAND) == FPL_PHOTO_URL.format(code=HAALAND)
    asyncio.run(run())


def test_cdn_errors_keep_the_fpl_photo_and_check_again(tmp_path):
    async def run():
        svc, _ = _service(tmp_path, cdn={CHERIF: httpx.ConnectError("down"), HAALAND: 503})
        await _loaded(svc)
        assert svc.peek("Sidiki Cherif") == FPL_PHOTO_URL.format(code=CHERIF)
        assert svc.peek("Erling Haaland") == FPL_PHOTO_URL.format(code=HAALAND)
        assert not svc._fpl_checked                           # retried on the next check
    asyncio.run(run())


def test_wikipedia_title_must_match_the_player():
    assert _title_matches("Sidiki Cherif", "Sidiki Cherif")
    assert _title_matches("Sidiki Chérif", "Sidiki Cherif")
    assert _title_matches("Nicolas Jackson (footballer)", "Nicolas Jackson")
    assert _title_matches("Ezri Konsa", "Ezri Konsa Ngoyo")
    assert _title_matches("Joelinton", "Joelinton")
    assert not _title_matches("Sidiki", "Sidiki Cherif")      # disambiguation page
    assert not _title_matches("Jackson (footballer)", "Nicolas Jackson")
    assert not _title_matches("Nicolas Pépé", "Nicolas Jackson")
