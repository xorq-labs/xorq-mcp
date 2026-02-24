"""Playwright end-to-end tests for the xorq web UI + Buckaroo integration.

These tests start real Buckaroo and xorq-web servers and verify the
full page rendering, iframe data loading, and error handling in a
headless browser.
"""

import json
from urllib.request import Request, urlopen

import pytest

from tests.conftest import BUCKAROO_TEST_PORT, WEB_TEST_PORT, _load_session

pytestmark = pytest.mark.playwright


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def web_url(path: str = "/") -> str:
    return f"http://localhost:{WEB_TEST_PORT}{path}"


def buckaroo_url(path: str = "/") -> str:
    return f"http://localhost:{BUCKAROO_TEST_PORT}{path}"


# ---------------------------------------------------------------------------
# Basic page rendering
# ---------------------------------------------------------------------------
class TestCatalogPage:
    def test_catalog_index_loads(self, page, web_server):
        page.goto(web_url("/"))
        page.wait_for_load_state("networkidle")
        assert "xorq" in page.title().lower() or page.locator(".sidebar").count() > 0

    def test_catalog_has_nav_entries(self, page, web_server):
        page.goto(web_url("/"))
        page.wait_for_load_state("networkidle")
        nav = page.locator(".nav-list")
        assert nav.count() >= 1

    def test_health_endpoint(self, page, web_server):
        page.goto(web_url("/health"))
        body = json.loads(page.locator("body").inner_text())
        assert body["status"] == "ok"


# ---------------------------------------------------------------------------
# Buckaroo iframe integration
# ---------------------------------------------------------------------------
class TestBuckarooIframe:
    def test_entry_page_has_iframe(self, page, web_server, buckaroo_server, small_parquet):
        """Load a dataset into Buckaroo and verify the iframe appears on the entry page."""
        # Pre-load data into Buckaroo so the handler can find it
        _load_session(small_parquet, "test_small", BUCKAROO_TEST_PORT)

        # Navigate to a catalog entry — we use a real entry if available
        page.goto(web_url("/"))
        page.wait_for_load_state("networkidle")

        # Find and click first entry link in the nav
        entry_links = page.locator(".nav-list a[href^='/entry/']")
        if entry_links.count() == 0:
            pytest.skip("No catalog entries available for iframe test")

        entry_links.first.click()
        page.wait_for_load_state("networkidle")

        # Check for either an iframe or an error message (both are valid outcomes)
        iframe = page.locator("iframe")
        error = page.locator(".buckaroo-error")
        assert iframe.count() > 0 or error.count() > 0, (
            "Expected either a Buckaroo iframe or an error message"
        )

    def test_buckaroo_session_page_renders(self, page, buckaroo_server, small_parquet):
        """Verify the Buckaroo standalone session page loads."""
        resp = _load_session(small_parquet, "pw_session", BUCKAROO_TEST_PORT)
        assert resp["session"] == "pw_session"
        assert resp["rows"] == 100

        page.goto(buckaroo_url("/s/pw_session"))
        page.wait_for_load_state("networkidle")
        # The page should have a root div and load standalone.js
        assert page.locator("#root").count() == 1

    def test_buckaroo_health(self, page, buckaroo_server):
        page.goto(buckaroo_url("/health"))
        body = json.loads(page.locator("body").inner_text())
        assert body["status"] == "ok"
        assert "pid" in body


# ---------------------------------------------------------------------------
# Large dataset stress tests
# ---------------------------------------------------------------------------
class TestLargeDatasets:
    def test_load_medium_dataset(self, buckaroo_server, medium_parquet):
        """100k rows should load without errors."""
        resp = _load_session(medium_parquet, "medium_ds", BUCKAROO_TEST_PORT)
        assert resp["rows"] == 100_000
        assert len(resp["columns"]) >= 4

    def test_load_large_dataset(self, buckaroo_server, large_parquet):
        """1M rows should load without crashing Buckaroo."""
        resp = _load_session(large_parquet, "large_ds", BUCKAROO_TEST_PORT)
        assert resp["rows"] == 1_000_000

        # Verify server is still healthy after loading large dataset
        health_resp = urlopen(buckaroo_url("/health"), timeout=5)
        health = json.loads(health_resp.read())
        assert health["status"] == "ok"

    def test_load_wide_dataset(self, buckaroo_server, wide_parquet):
        """1000 columns should load without errors."""
        resp = _load_session(wide_parquet, "wide_ds", BUCKAROO_TEST_PORT)
        assert resp["rows"] == 10_000
        assert len(resp["columns"]) == 1000

    def test_large_dataset_page_renders(self, page, buckaroo_server, large_parquet):
        """The session page should render for a 1M-row dataset without timing out."""
        _load_session(large_parquet, "large_page", BUCKAROO_TEST_PORT)
        page.goto(buckaroo_url("/s/large_page"))
        # Give it up to 15 seconds for a large dataset
        page.wait_for_load_state("networkidle", timeout=15_000)
        assert page.locator("#root").count() == 1


# ---------------------------------------------------------------------------
# Session reload / stale data
# ---------------------------------------------------------------------------
class TestSessionReload:
    def test_reload_session_with_new_data(self, buckaroo_server, small_parquet, tmp_path):
        """Loading the same session_id with different data should update it."""
        # Load initial data
        resp1 = _load_session(small_parquet, "reload_test", BUCKAROO_TEST_PORT)
        assert resp1["rows"] == 100

        # Create different data
        import pandas as pd

        df2 = pd.DataFrame({"x": range(50)})
        path2 = tmp_path / "small2.parquet"
        df2.to_parquet(str(path2))

        # Reload same session with new data
        resp2 = _load_session(str(path2), "reload_test", BUCKAROO_TEST_PORT)
        assert resp2["rows"] == 50, (
            f"Expected 50 rows after reload, got {resp2['rows']}. Session data was not refreshed."
        )

    def test_rapid_session_creation(self, buckaroo_server, small_parquet):
        """Create many sessions rapidly — server should not crash."""
        for i in range(20):
            resp = _load_session(small_parquet, f"rapid_{i}", BUCKAROO_TEST_PORT)
            assert resp["rows"] == 100

        # Server should still be healthy
        health = json.loads(urlopen(buckaroo_url("/health"), timeout=5).read())
        assert health["status"] == "ok"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
class TestErrorHandling:
    def test_missing_file_returns_404(self, buckaroo_server):
        """Loading a nonexistent file should return 404, not crash."""
        payload = json.dumps(
            {"session": "missing", "path": "/nonexistent/file.parquet", "mode": "lazy"}
        ).encode()
        req = Request(
            buckaroo_url("/load"),
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        from urllib.error import HTTPError

        with pytest.raises(HTTPError) as exc_info:
            urlopen(req, timeout=10)
        assert exc_info.value.code == 404

    def test_invalid_json_returns_400(self, buckaroo_server):
        """Posting garbage JSON should return 400."""
        req = Request(
            buckaroo_url("/load"),
            data=b"not json{{{",
            headers={"Content-Type": "application/json"},
        )
        from urllib.error import HTTPError

        with pytest.raises(HTTPError) as exc_info:
            urlopen(req, timeout=10)
        assert exc_info.value.code == 400

    def test_nonexistent_session_page(self, page, buckaroo_server):
        """Visiting a session that was never created should still return HTML."""
        page.goto(buckaroo_url("/s/does_not_exist"))
        # Should not crash — the page renders but WS won't have data
        assert page.locator("#root").count() == 1

    def test_web_404_for_missing_entry(self, page, web_server):
        """The web UI should return 404 for a nonexistent catalog entry."""
        resp = page.goto(web_url("/entry/definitely_does_not_exist_12345"))
        assert resp.status == 404

    def test_buckaroo_error_displayed_in_web_ui(self, page, web_server):
        """When Buckaroo fails to load an expression, the error message
        should be visible in the page (not silently hidden)."""
        page.goto(web_url("/"))
        page.wait_for_load_state("networkidle")

        # Navigate to a catalog entry
        entry_links = page.locator(".nav-list a[href^='/entry/']")
        if entry_links.count() == 0:
            pytest.skip("No catalog entries for error display test")

        entry_links.first.click()
        page.wait_for_load_state("networkidle")

        # At this point, either the iframe rendered or the error is shown.
        # Both are acceptable — the key is that the page isn't blank.
        iframe = page.locator("iframe")
        error = page.locator(".buckaroo-error")
        data_preview = page.locator("text=Data Preview")
        metadata = page.locator("text=Build Metadata")

        # The page should have at least the metadata section
        assert metadata.count() > 0, "Build Metadata section missing"

        # And either an iframe, an error message, or at least the section title
        has_content = iframe.count() > 0 or error.count() > 0 or data_preview.count() > 0
        assert has_content, "Data Preview section completely missing with no error"


# ---------------------------------------------------------------------------
# Rapid navigation (reproduces flaky iframe loading)
# ---------------------------------------------------------------------------
class TestRapidNavigation:
    def test_quick_nav_between_entries(self, page, web_server):
        """Rapidly clicking between different catalog entries should not crash."""
        page.goto(web_url("/"))
        page.wait_for_load_state("networkidle")

        entry_links = page.locator(".nav-list a[href^='/entry/']")
        count = entry_links.count()
        if count < 2:
            pytest.skip("Need at least 2 catalog entries for navigation test")

        # Rapidly click through entries
        for _ in range(3):
            for i in range(min(count, 5)):
                entry_links.nth(i).click()
                # Don't wait for full load — simulate impatient clicking
                page.wait_for_load_state("domcontentloaded", timeout=5_000)

        # After all that clicking, the page should still be functional
        page.wait_for_load_state("networkidle", timeout=15_000)
        assert page.locator(".sidebar").count() > 0
