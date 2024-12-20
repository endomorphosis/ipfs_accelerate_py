#capture args from command line
import sys
import json
import base64
from tkinter import N
from PIL import Image
from io import BytesIO
import struct
import json
import requests
from PIL import Image
from io import BytesIO

class ClipRetrieval:
	def __init__(self, resources, meta=None):
		if meta is not None:
			if "config" in meta:
				if meta['config'] is not None:
					self.config = meta['config']
		if "commands" in resources:
			self.commands = resources["commands"]

		if "skills" in resources:
			self.skills = resources["skills"]

	def __call__(self, method, **kwargs):
		if method == 'retrieve_image':
			self.method = 'retrieve_image'
			return self.retrieve_image(**kwargs)
		pass

	def retrieve_image(self, **kwargs):

		return None

	def retrieve_image_from_url(self, url):
		try:
			response = requests.get(url)
			#print(response.status_code)
			if response.status_code == 200:
				return Image.open(BytesIO(response.content))
			if response.status_code == 404:
				return None
			else:
				return None
		except:
			return None
		

	def resize_image(self, image, size):
		return image.resize(size, Image.Resampling.LANCZOS)

	def write_image_to_file(self, image, filename):
		image.save(filename)

	def detect_image_size(self, image):
		return image.size

	#capture args from command line
	#check that sys.argv is not empty
	def main(self, **kwargs):
		if len(sys.argv) < 3:
			print("Error: empty args")
			sys.exit(1)

		else:
			#capture args from command line
			endpoint = sys.argv[1]
			indice = sys.argv[2]
			query_text = sys.argv[3]
			timestamp = sys.argv[4]
			x = int(sys.argv[5])
			y = int(sys.argv[6])
			from clip_retrieval.clip_client import ClipClient, Modality
			client = ClipClient(url=endpoint, indice_name=indice)
			results = client.query(text=query_text)
			#print(len(results))
			#print(results[0])
			new_results = []

			for result in results:
				#print(result)
				image = retrieve_image_from_url(result["url"])
				if image is not None:
					size = detect_image_size(image)
					image_ratio = size[0]/size[1]
					if image_ratio > 1:
						image_ratio = 1/image_ratio
						result["size"] = size
						result["image_ratio"] = image_ratio
						new_results.append(result)
				#print("url "+ result["url"])
				#print("size ")
				#print(size)
				#result["image"] = image
			image_ratio = float(int(x)/int(y))

			sort_results = sorted(new_results, key=lambda k: k['image_ratio'], reverse=True)

			#sort by the the closest image ratio to image_ratio in sort_results
			sort_results = sorted(sort_results, key=lambda k: abs(k['image_ratio'] - image_ratio), reverse=False)

			#print(sort_results)
			for result in sort_results:
				#print("size")
				#print(result["size"])
				if result["size"][0] > 256 and result["size"][1] > 256:
					image = retrieve_image_from_url(result["url"])
					image = resize_image(image, (x,y))	
					write_image_to_file(image, "./samples/"+ timestamp+"_src.png")

					#print(result)
					sys.stdout.write(json.dumps(result))
					#results to stdout
					break


			#sort image_ratio by descending order

			#resized_image = resize_image(image, (512, 512))
		



#> {'url': 'https://example.com/kitten.jpg', 'caption': 'an image of a kitten', 'id': 14, 'similarity': 0.2367108941078186}
