import pytest
from server.url_processor import clean_content


# ── clean_content ────────────────────────────────────────────────────────────

def test_clean_content_removes_images():
    raw = "开头\n\n![cover_image](https://mmbiz.qpic.cn/xxx)\n\n正文内容"
    assert "![cover_image]" not in clean_content(raw)
    assert "正文内容" in clean_content(raw)


def test_clean_content_removes_empty_headings():
    raw = "#\n\n##\n\n###\n\n正文"
    result = clean_content(raw)
    assert "#\n" not in result
    assert "正文" in result


def test_clean_content_removes_separators():
    raw = "上面\n\n---\n\n下面"
    result = clean_content(raw)
    assert "---" not in result
    assert "上面" in result
    assert "下面" in result


def test_clean_content_collapses_blank_lines():
    raw = "第一行\n\n\n\n\n\n第二行"
    result = clean_content(raw)
    assert result == "第一行\n\n第二行"


def test_clean_content_strips_whitespace():
    raw = "  \n内容\n  \n"
    assert clean_content(raw) == "内容"


def test_clean_content_removes_image_variants():
    raw = "![图1](url1) 文字 ![图2](url2) 结尾"
    result = clean_content(raw)
    assert "![图1]" not in result
    assert "![图2]" not in result
    assert "文字" in result
    assert "结尾" in result


# ── fetch_url_text ────────────────────────────────────────────────────────────

def test_fetch_url_text_static_success(mocker):
    """静态页面提取成功，trafilatura 返回内容 > 200 chars。"""
    mocker.patch('trafilatura.fetch_url', return_value='<html><body><p>静态内容正文...（超过200字）'
                                                       '</p></body></html>')
    mocker.patch('trafilatura.extract', return_value='静态内容正文。' * 50)
    from server.url_processor import fetch_url_text
    result = fetch_url_text('https://example.com')
    assert result['url'] == 'https://example.com'
    assert len(result['content']) > 200
    assert result['title'] is None or isinstance(result['title'], str)


def test_fetch_url_text_fallback_to_tavily(mocker):
    """trafilatura 提取内容不足 200 chars → 降级到 tavily。"""
    mocker.patch('trafilatura.fetch_url', return_value='<html></html>')
    mocker.patch('trafilatura.extract', return_value='短内容')
    mock_tavily = mocker.patch('server.url_processor.TavilyClient')
    mock_tavily.return_value.extract.return_value = {
        'results': [{'content': 'Tavily 提取的长篇正文内容' * 50}]
    }
    from server.url_processor import fetch_url_text
    result = fetch_url_text('https://example.com')
    assert 'Tavily' in result['content']
    assert len(result['content']) > 200


def test_fetch_url_text_tavily_returns_none(mocker):
    """trafilatura 和 tavily 都失败时返回错误信息。"""
    mocker.patch('trafilatura.fetch_url', return_value=None)
    mocker.patch('trafilatura.extract', return_value='')
    mock_tavily = mocker.patch('server.url_processor.TavilyClient')
    mock_tavily.return_value.extract.return_value = {'results': []}
    mock_resp = mocker.Mock()
    mock_resp.status_code = 200
    mock_resp.text = '<html><head><title></title></html>'
    mocker.patch('requests.get', return_value=mock_resp)
    from server.url_processor import fetch_url_text
    result = fetch_url_text('https://example.com')
    assert result['content'] == '[抓取失败]'
    assert result['title'] is None


def test_fetch_url_text_extracts_title(mocker):
    """从 HTML <title> 标签提取标题。"""
    mock_resp = mocker.Mock()
    mock_resp.status_code = 200
    mock_resp.text = '<html><head><title>测试文章标题</title></head><body><p>内容</p></body></html>'
    mocker.patch('requests.get', return_value=mock_resp)
    mocker.patch('trafilatura.fetch_url',
                 return_value='<html><head><title>测试文章标题</title></head><body><p>内容</p></body></html>')
    mocker.patch('trafilatura.extract', return_value='内容内容' * 50)
    from server.url_processor import fetch_url_text
    result = fetch_url_text('https://example.com')
    assert result['title'] == '测试文章标题'


def test_fetch_url_text_timeout(mocker):
    """网络超时场景。"""
    mocker.patch('trafilatura.fetch_url', side_effect=Exception('Timeout'))
    from server.url_processor import fetch_url_text
    result = fetch_url_text('https://example.com')
    assert result['content'] == '[抓取失败]'


# ── fetch_urls_concurrent ────────────────────────────────────────────────────

def test_fetch_urls_concurrent_all_success(mocker):
    """多个 URL 全部成功。"""
    mocker.patch('server.url_processor.fetch_url_text', side_effect=[
        {"url": "https://a.com", "title": "A", "content": "内容A"},
        {"url": "https://b.com", "title": "B", "content": "内容B"},
    ])
    from server.url_processor import fetch_urls_concurrent
    results = fetch_urls_concurrent(["https://a.com", "https://b.com"])
    assert len(results) == 2
    assert results[0]["title"] == "A"
    assert results[1]["title"] == "B"


def test_fetch_urls_concurrent_partial_failure(mocker):
    """部分 URL 失败，不影响其他。"""
    mocker.patch('server.url_processor.fetch_url_text', side_effect=[
        {"url": "https://a.com", "title": "A", "content": "内容A"},
        {"url": "https://b.com", "title": None, "content": "[抓取失败]"},
        {"url": "https://c.com", "title": "C", "content": "内容C"},
    ])
    from server.url_processor import fetch_urls_concurrent
    results = fetch_urls_concurrent(["https://a.com", "https://b.com", "https://c.com"])
    assert len(results) == 3
    assert results[0]["content"] == "内容A"
    assert results[1]["content"] == "[抓取失败]"
    assert results[2]["content"] == "内容C"


def test_fetch_urls_concurrent_empty():
    """空列表返回空列表。"""
    from server.url_processor import fetch_urls_concurrent
    assert fetch_urls_concurrent([]) == []


def test_fetch_urls_concurrent_deduplicates(mocker):
    """重复 URL 只抓取一次。"""
    mock_fn = mocker.patch('server.url_processor.fetch_url_text', return_value={
        "url": "https://a.com", "title": "A", "content": "内容A"
    })
    from server.url_processor import fetch_urls_concurrent
    results = fetch_urls_concurrent(["https://a.com", "https://a.com"])
    assert len(results) == 1
    mock_fn.assert_called_once()
