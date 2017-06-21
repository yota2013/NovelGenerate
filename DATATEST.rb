#! /usr/bin/ruby
# coding: UTF-8
require 'nokogiri'
require 'rubygems'
require 'amazon/ecs'

Amazon::Ecs.debug = true
Amazon::Ecs.options = {
  :associate_tag =>     '',
  :AWS_access_key_id => '',
  :AWS_secret_key =>    ''
}

#item_search の第一引数で読み方を記入
# response_group は，'Small', 'Medium', 'Large' の 3種　ItemAttributesは商品情報が
page = 1
while true do
  res = Amazon::Ecs.item_search('ラノベ', {:search_index => 'Books', :response_group => 'Medium,ItemAttributes', :browse_node => '466280', :country => 'jp',:item_page => 12})
  puts(res.total_pages)
  puts(page)
  res.items.each do |item|
  #puts item.get_element('ItemAttributes')

  #階層が下のものも取得できる
  puts "Title:#{item.get('ItemAttributes/Title')}"
  #puts "Author:#{item.get('ItemAttributes/Author')}"
  end
  # 画像のurlとサイズをhashで取得
  #puts item.get_hash('SmallImage')
  sleep(9)
  page = page + 1
  if res.total_pages < page then
   break
  end

end
