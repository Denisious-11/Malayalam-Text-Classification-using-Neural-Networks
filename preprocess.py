adjectives="Collections/adjectives.txt"
pronouns="Collections/pronouns.txt"
postpositions="Collections/postpositions.txt"
negations="Collections/negations.txt"
interjections="Collections/interjections.txt"
demonstratives="Collections/demonstratives.txt"
conjuctions="Collections/conjunctions.txt"
affirmatives="Collections/affirmatives.txt"

with open(adjectives,encoding="utf8") as f:
	content_list1 = f.readlines()
cont_adjectives = [x.strip() for x in content_list1]# remove new line characters


with open(pronouns,encoding="utf8") as f:
    content_list2 = f.readlines()

# remove new line characters
cont_pronouns = [x.strip() for x in content_list2]


with open(postpositions,encoding="utf8") as f:
    content_list3 = f.readlines()

# remove new line characters
cont_postpositions = [x.strip() for x in content_list3]


with open(negations,encoding="utf8") as f:
    content_list4 = f.readlines()

# remove new line characters
cont_negations = [x.strip() for x in content_list4]


with open(interjections,encoding="utf8") as f:
    content_list5 = f.readlines()

# remove new line characters
cont_interjections = [x.strip() for x in content_list5]


with open(demonstratives,encoding="utf8") as f:
    content_list6 = f.readlines()

# remove new line characters
cont_demonstratives = [x.strip() for x in content_list6]


with open(conjuctions,encoding="utf8") as f:
    content_list7 = f.readlines()

# remove new line characters
cont_conjuctions = [x.strip() for x in content_list7]


with open(affirmatives,encoding="utf8") as f:
    content_list8 = f.readlines()

# remove new line characters
cont_affirmatives = [x.strip() for x in content_list8]



def remove_adjectives(txt):
	tokens = txt.split()
	tokens_filtered= [word for word in tokens if not word in cont_adjectives]
	return (" ").join(tokens_filtered)

def remove_pronouns(txt):
	tokens = txt.split()
	tokens_filtered= [word for word in tokens if not word in cont_pronouns]
	return (" ").join(tokens_filtered)

def remove_postpositions(txt):
	tokens = txt.split()
	tokens_filtered= [word for word in tokens if not word in cont_postpositions]
	return (" ").join(tokens_filtered)

def remove_negations(txt):
	tokens = txt.split()
	tokens_filtered= [word for word in tokens if not word in cont_negations]
	return (" ").join(tokens_filtered)

def remove_interjections(txt):
	tokens = txt.split()
	tokens_filtered= [word for word in tokens if not word in cont_interjections]
	return (" ").join(tokens_filtered)

def remove_demonstratives(txt):
	tokens = txt.split()
	tokens_filtered= [word for word in tokens if not word in cont_demonstratives]
	return (" ").join(tokens_filtered)

def remove_conjuctions(txt):
	tokens = txt.split()
	tokens_filtered= [word for word in tokens if not word in cont_conjuctions]
	return (" ").join(tokens_filtered)

def remove_affirmatives(txt):
	tokens = txt.split()
	tokens_filtered= [word for word in tokens if not word in cont_affirmatives]
	return (" ").join(tokens_filtered)


def cleantext(txt):
	txt1=remove_adjectives(txt)
	txt2=remove_pronouns(txt1)
	txt3=remove_postpositions(txt2)
	txt4=remove_negations(txt3)
	txt5=remove_interjections(txt4)
	txt6=remove_demonstratives(txt5)
	txt7=remove_conjuctions(txt6)
	txt8=remove_affirmatives(txt7)

	return txt8


