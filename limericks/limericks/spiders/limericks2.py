import scrapy
import re

class LimerickItem(scrapy.Item):
    # define the fields for your item here like:
    limerick=scrapy.Field()
    pass

def strip_list(my_list): #takes a list of html code extracts the underlying sting and concatenates
  t=''
  for elt in my_list:
    t+=re.sub('<.*?>','',elt)
  return t
    
class LimerickSpider(scrapy.Spider): # a spider for scraping limericks
  name = "limerick2"
  allowed_domains = ["oedilf.com"]
  start_urls = ['http://www.oedilf.com/db/Lim.php?Show=Authors']
   
  def parse(self,response): #first get the links for each author
    items=[]
    l=response.xpath('.//a[@title="RSS feed"]/@href').extract() # the ones who have actually posted a limericks has an RSS feed
    author_ids=[]
    for item in l:
      author_ids.append(item[12:-7]) #this is the id number from the url
    author_urls=[]
    for aid in author_ids:
        author_urls.append('http://www.oedilf.com/db/Lim.php?AuthorId='+ aid)
    for link in author_urls:
        request=scrapy.Request(link,callback=self.parse_layer2)
        yield request
  def parse_layer2(self,response): # this layer figures out how many limericks the author has
      try:
          last=response.xpath('.//div[@id="content"]/a/@href')[-1].extract() #this almost (+- 10) of the number of limericks the author has written
          m=re.search('Start=.*',last)
          bound=int(m.group()[6:])
      except IndexError:
          bound=1
      ix=0
      while ix<=bound:
          link=response.url+'&Start=' +str(ix) #each of these pages has <=10 limericks
          request=scrapy.Request(link,callback=self.parse_layer3)
          ix+=10
          yield request     
  def parse_layer3(self,response): #navigate to the limericks page and extract the limericks on the page
      items=[]
      lims=response.xpath('.//div[@id="content"]//div[@class="limerickverse"]')# the limericks on the page
      for lim in lims:#format each limerick as an ascii string
          item=LimerickItem()
          s=lim.xpath('./node()').extract()
          t=strip_list(s)
          t=t.encode('ascii','replace').lower()+'\r\n'
          item['limerick']=t
          items.append(item)
      return items      