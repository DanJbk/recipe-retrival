from bs4 import BeautifulSoup as bs
import urllib


class html_scanner:
    @staticmethod
    def fetch_text(url):
        html = urllib.request.urlopen(url).read().decode("utf-8")
        bso = bs(html, "html.parser")

        return html_scanner.list_crawler(bso) + html_scanner.text_crawler(bso)

    @staticmethod
    def text_crawler(bso):
        texts = []
        for p_tag in bso.find_all(["p"]):
            texts.append(p_tag.text)

        return [text for text in texts if len(text) > 1]

    @staticmethod
    def list_crawler(bso):
        texts = []
        for ul_tag in bso.find_all(["ul"]):

            a_list = []
            for content in ul_tag.find_all("li"):
                if content not in ["", "\n"]:
                    a_list.append(content.text)

            texts.append("\n".join(a_list))

        return texts
