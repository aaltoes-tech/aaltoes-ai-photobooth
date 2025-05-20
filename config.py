prompt_text = """This photo is taken at a photo booth and has a group of people. 
    There can be some people in the background,ignore them in the explanation. 
    Mention the number of people who are taking photo. Expain who is at the 
    picture: gender, age, explain appearance (mention hair color), 
    always include the action they are doing for each person individually. 
    What is the body type the have slim/normal/a bit chubby.  Be as precise as 
    possible. Do not mention background. Do not mention clothes of person. 
    Use at most 30 words per each person. Do not mention my instructions. Mention what 
    hair would suit for each person in the style of 
"""

hair_prompt = ["Explorer, suggest if colorful highlights on hair would suit this style and person.  Source of light is from the left and it reflects from the hair. Anatomically correct. All fingers have corect anatomical form.",
            "Inventor, suggest if colorful highlights on hair would suit this style and person.  Source of light is from the left and it reflects from the hair. Anatomically correct. All fingers have corect anatomical form.",
            "Scholar, suggest if colorful highlights on hair would suit this style and person.  Source of light is from the left and it reflects from the hair. Anatomically correct. All fingers have corect anatomical form.",
            "Scientist, suggest if colorful highlights on hair would suit this style and person.  Source of light is from the left and it reflects from the hair. Anatomically correct. All fingers have corect anatomical form."
]

clothes_prompt = [
    "Clothes: Sapphire-violet pioneer gear: insulated ensemble with hidden pockets, hex-weave moisture-wicking, constellation accents, glowing data canisters.",
    "Clothes: Navy-slate workshop attire: reinforced, concealed-tool outfit with hex-knit moisture-wicking fabric, utility belt, oil-stained phosphor seams.",
    "Clothes: Navy-charcoal-violet explorer regalia: matte tri-tone ensemble with sigil etchings, microLED mesh, utility straps, luminescent embroidery.",
    "Clothes: White-graphite labwear: weatherproof layers with hex-weave underlayers, hidden compartments, modular belt for data slates, micro-LED accents."
]

background_prompt = [
    "Background: Aqua-violet mist corridor: liquid-starlight walls, floating bioluminescent motes, blurred haze, fading into a dark point of promise.",
    "Background: Cool-blue invention cathedral: vaulted steel rafters above gear-strewn benches, pipe pillars entwined with lab instruments; frosted windows casting prismatic halos; drifting stardust motes.",
    "Background: Starfield bleeding into chalk-scrawled pillars; marble steps rising into lavender fog; floating shelves and ghostly tomes amid starlight motes.",
    "Background: Midnight lab in white and violet-blue: algae-streaked walls, humming consoles, a rippling code wall; a lone case illuminating calibrators and notebooks."
]


replicate_model = "ideogram-ai/ideogram-v3-turbo"