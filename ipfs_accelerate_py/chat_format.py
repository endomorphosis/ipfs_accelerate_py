import os
import yaml

class chat_format:
	def __init__(self, resources, meta=None):
		with open(os.path.join(os.path.dirname(__file__), 'templates.yml')) as f:
			self.templates = yaml.load(f, Loader=yaml.Loader)

	def format_chat_prompt(self, template, messages):
		templates = self.templates

		if type(template) == dict:
			if 'with_system' not in template or 'without_system' not in template:
				raise Exception('Custom templates must define "with_system" and "without_system"')
		else:
			if template not in templates:
				raise Exception('No chat template for "%s" defined' % template)
			else:
				template = templates[template]
			
		if len(messages) == 0:
			raise Exception('No messages passed (message list is empty)')
		attempts = {}
		for template_name in list(templates.keys()):
			this_template = templates[template_name]
			try:
				attempts[template_name] = self.split_messages(this_template, messages)[1]
			except Exception as e:
				attempts[template_name] = str(e)

		detected_template = None
		most_messages = 0
		for attempt in attempts:
			this_attempt = attempts[attempt]
			if type(this_attempt) == list:
				if len(attempts[attempt]) > most_messages and len(attempts[attempt]) > 0:
					detected_template = attempt
					most_messages = len(attempts[attempt])


		system, rounds = self.split_messages(templates[detected_template], messages)
		blocks = []

		if not system and 'system_default' in template:
			system = template['system_default']

		for i in range(len(rounds)):
			is_first = i == 0
			is_last = i >= len(rounds) - 1
			user, assistant = rounds[i]

			blocks.append(
				self.format_round(
					template, 
					system=system if is_first else None,
					user=user,
					assistant=assistant,
					closed=not is_last
				)
			)

		final_prompt = (
			template['round_seperator']
			if 'round_seperator' in template 
			else ''
		).join(blocks)

		if 'stop' in template:
			stop = template['stop']
		else:
			stop = template['without_system'].split('{assistant}')[1]

			if len(stop) == 0:
				stop = None

		return final_prompt, stop, detected_template


	def split_messages(self, template, messages):
		rounds = []
		system = None
		user_messages = []
		assistant_messages = []
		system_messages = []
		if type (messages) == str:
			if "{user" in template['with_system'] or '{assistant' in template['with_system']:
			
				if "{"+"system}" in template['with_system']:
					index_system_end = template['with_system'].index("{"+"system}") + len("{"+"system}")
					system_prefix = template['with_system'][:index_system_end].split("{"+"system}")[0]
					index_system_prefix_end = template['with_system'].index(system_prefix)
				
				if "{"+"user}" in template['with_system'] or "assistant}" in template['with_system']:
					index_user_end = template['with_system'].index("{"+"user}")
					user_prefix = template['with_system'][index_system_end:index_user_end].split("{"+"user}")[0].replace(template["round_seperator"], "")
					index_assistant_end = template['with_system'].index("{"+"assistant}") + len("{"+"assistant}")
					assistant_suffix = template['with_system'][index_assistant_end:]
					assistant_prefix = template['with_system'][index_user_end:index_assistant_end].split("{"+"assistant}")[0].replace(template["round_seperator"], "").replace("{"+"user}", "").replace(assistant_suffix, "")
					user_suffix = template['with_system'][index_user_end:index_assistant_end].replace("{"+"user}", "").replace(template["round_seperator"], "").replace(assistant_prefix, "").replace("{"+"assistant}", "")
					user_suffix = template['with_system'][index_user_end:index_assistant_end - len(assistant_prefix) - len("{"+"assistant}")].replace("{"+"user}", "").replace(template["round_seperator"], "")
					assistant_prefix = assistant_prefix.replace(user_suffix, "").replace("{"+"user}", "")

				if "system}" in template['with_system']:
					system_suffix = template['with_system'][index_system_prefix_end:index_user_end - len(user_prefix)].replace("{"+"system}", "").replace(template["round_seperator"], "")

				system_exists = messages.find(system_prefix) != -1 and messages.find(system_suffix) != -1 and messages.find(user_prefix) != 0 and messages.find(assistant_prefix) != 0
 
				if system_exists:
					system_message_index = messages.find(system_prefix) 
					system = messages[messages.find(system_prefix) + len(system_prefix):messages.find(system_suffix)]
					system_message = messages.replace(system_prefix, "").replace(system_suffix, "")
					messages = messages.replace(system_prefix, "").replace(system_suffix, "").replace(system_message, "")

				message_list = messages.split(template["round_seperator"])
				message_list = [message for message in message_list if message != ""]
				for message in message_list:
					if message.find(user_prefix) != -1 and message.find(user_suffix) != -1 and message.find("</s>") != -1:
						user_messages.append(message[message.find(user_prefix) + len(user_prefix):message.find(user_suffix)])
					if message.find(assistant_prefix) != -1 and message.find(assistant_suffix) != -1 and message.find("</s>") != -1:
						assistant_messages.append(message[message.find(assistant_prefix) + len(assistant_prefix):message.find(assistant_suffix)])
					
				message_rounds = len(user_messages) if len(user_messages) > len(assistant_messages) else len(assistant_messages)
				first_message = "system" if system_exists else "user" if messages.find(user_prefix) == 0 else "assistant" if messages.find(assistant_prefix) == 0 else None
				len_user_messages = len(user_messages)
				len_assistant_messages = len(assistant_messages)

				for i in range(message_rounds):
					if first_message == "user" and i == 0:
						rounds.append((user_messages[i], assistant_messages[i]))
					elif first_message == "assistant" and i == 0:
						user_messages.insert(0, "")
						rounds.append((user_messages[i], assistant_messages[i]))
					else:
						if (i < len(user_messages)) and (i < len(assistant_messages)):
							rounds.append((user_messages[i], assistant_messages[i]))
						elif i < len(user_messages) and i >= len(assistant_messages):
							rounds.append((user_messages[i], ""))
						elif i >= len(user_messages) and i < len(assistant_messages):
							rounds.append(("", assistant_messages[i]))

		if type(messages) == dict:
			if messages[0]['role'] == 'system':
				system = messages[0]['content']
				offset = 1
			else:
				system = None
				offset = 0

			for i in range(offset, len(messages), 2):
				m1 = messages[i]
				m2 = messages[i+1] if len(messages) >= i+2 else None


				if m1['role'] != 'user':
					raise Exception('Message #%i must be of role "user"' % (i+1))
				
				if m2 and m2['role'] != 'assistant':
					raise Exception('Message #%i must be of role "assistant"' % (i+2))
				
				rounds.append((m1['content'], m2['content'] if m2 else None))

		return system, rounds

	def format_round(self, template, system, user, assistant, closed=True):
		if system:
			prompt = template['with_system']
			prompt = prompt.replace('{system}', system)
		else:
			prompt = template['without_system']

		prompt = prompt.replace('{user}', user)

		if closed == False:
			prompt = prompt.replace('{assistant}', assistant)
		else:
			index = prompt.index('{assistant}')
			
			start = prompt[:index]
			end = prompt[index:].replace("{assistant}", "").replace(" ", "")
			if not "}" in end:
				prompt = start
				if assistant:
					prompt += assistant
			else:
				prompt = prompt.replace('{assistant}', assistant)
				#prompt = prompt.replace(template["round_seperator"], "")

		return prompt

	
if __name__ == "__main__":
	this_chat_format = chat_format(None)
	chat_template = "openai"
	messages_without_images = 'ASSISTANT: How can I help you today?</s> \nUSER: I would like to know the weather forecast for tomorrow in Berlin. What do you think?</s> \nASSISTANT: The weather will be fine, sunny and warm. Enjoy your stay in Berlin!</s> \nUSER: Thank you very much. Goodbye.</s> \nASSISTANT: You are welcome. Goodbye!</s> \n\\end{code}\n\\begin{itemize}\n\\item The `translate` function is used to translate the English text into German text. This is done by using the `nlp_model` function from the `spacy` library, which uses a pre-trained model trained on a large corpus of English text. The `nlp_model` function takes an input string and returns the output string translated to German. In this case, we are translating the entire input string into German. This is done by using the `nlp_model` function from the `spacy` library, which uses a pre-trained model trained on a large corpus of English text. The `nlp_model` function takes an input string and returns the'
	prompt = this_chat_format.format_chat_prompt(chat_template, messages_without_images)
	print(prompt)
	pass