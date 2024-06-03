import os
import requests
import fitz  
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from bs4 import BeautifulSoup

def init_webdriver():
    options = Options()
    options.add_argument('--headless') 
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    driver = webdriver.Chrome(options=options)
    return driver

def download_and_extract_pdf_text(pdf_url):
    response = requests.get(pdf_url)
    pdf_path = 'temp.pdf'
    with open(pdf_path, 'wb') as f:
        f.write(response.content)
    pdf_text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            pdf_text += page.get_text()
    os.remove(pdf_path)
    return pdf_text

def extract_text_from_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text(separator=' ', strip=True)

def collect_text_elements(url, driver):
    driver.get(url)
    time.sleep(5) 
    collected_text = []
    body = driver.find_element(By.TAG_NAME, 'body')
    divs = body.find_elements(By.TAG_NAME, 'div')
    for div in divs:
        try:
            html = div.get_attribute('innerHTML')
            text = extract_text_from_html(html)
            if text:
                collected_text.append(text)
        except:
            print(url)

    links = body.find_elements(By.TAG_NAME, 'a')
    subpage_urls = []
    for link in links:
        href = link.get_attribute('href')
        if href and not should_skip_link(link):
            if href.endswith('.pdf'):
            
                pdf_text = download_and_extract_pdf_text(href)
                collected_text.append(pdf_text)
            else:
                subpage_urls.append(href)
    return collected_text, subpage_urls

def should_skip_link(link):
    skip_keywords = ['contact', 'login', 'sign up', 'signup', 'chat', 'language', 'lang']
    link_text = link.text.lower()
    link_href = link.get_attribute('href').lower()
    for keyword in skip_keywords:
        if keyword in link_text or keyword in link_href:
            return True
    if 'u.ae' not in link_href:
        return True
    return False

def recursive_collect_text(url, driver, visited_urls=None):
    if visited_urls is not None:
        print(len(visited_urls))
    if visited_urls is None:
        visited_urls = set()
    if len(visited_urls) >= 150:
        return {}
    if url in visited_urls:
        return {}
    visited_urls.add(url)
    collected_text, subpage_urls = collect_text_elements(url, driver)
    all_collected_text = {url: collected_text}
    for subpage_url in subpage_urls:
        if subpage_url not in visited_urls and len(visited_urls) < 150:
            subpage_text = recursive_collect_text(subpage_url, driver, visited_urls)
            all_collected_text.update(subpage_text)
    return all_collected_text

def main():
    start_url = 'https://u.ae/en/resources/faqs'
    driver = init_webdriver() 
    try:
        collected_text = recursive_collect_text(start_url, driver)
    finally:
        driver.quit()
    with open('collected_text.txt', 'w', encoding='utf-8') as file:
        for url, texts in collected_text.items():
            file.write(f"{url}\n")
            for text in texts:
                file.write(f"{text}\n")
            file.write("\n")

if __name__ == "__main__":
    main()
