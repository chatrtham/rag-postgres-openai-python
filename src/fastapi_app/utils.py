import re


def update_urls_with_utm(content: str, pattern: str, utm_source: str = "ai-chat") -> str:
    urls = re.findall(pattern, content)
    updated_urls = [add_utm_param(url, utm_source) for url in urls]

    for old_url, new_url in zip(urls, updated_urls):
        content = content.replace(old_url, new_url)
    
    return content

def add_utm_param(url: str, utm_source: str = "ai-chat") -> str:
    if "?" in url:
        return f"{url}&utm_source={utm_source}"
    else:
        return f"{url}?utm_source={utm_source}"