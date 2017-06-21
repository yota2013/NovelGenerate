#! /usr/bin/ruby
# coding: UTF-8
require 'rubygems'
require 'nokogiri'
require 'open-uri'

page = 2
#genres = [101,102,201,202,301,302,303,304,305,306,307,401,403,9901,9902,9903,9904,9999,9801]
genres = [302,303,304,305,306,307,401,403,9901,9902,9903,9904,9999,9801]

for genre in genres do
  f = File.open("NovelTile#{genre}.txt", "a")
  url = "http://yomou.syosetu.com/search.php?notnizi=1&word=&notword=&genre=#{genre}&order=hyoka&type="
  doc = Nokogiri::HTML(open(url))
  doc.css('.novel_h').each do |element|
      puts element.inner_text #テキスト
      f.puts(element.inner_text)
  end

  while true do
    pagechange_url="http://yomou.syosetu.com/search.php?&order=hyoka&notnizi=1&genre=#{genre}&p=#{page}"
    doc = Nokogiri::HTML(open(pagechange_url))
    doc.css('.novel_h').each do |element|
        puts element.inner_text #テキスト
        f.puts(element.inner_text)
    end
    page = page + 1
    if 100 < page then
     break
    end
    sleep(1)
  end
  page = 2
end


#doc = Nokogiri::HTML(open(pagechange_url))
