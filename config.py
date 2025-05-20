prompt_text = """This photo is taken at a photo booth and has a group of people. 
    There can be some people in the background,ignore them in the explanation. 
    Mention the number of people who are taking photo. Expain who is at the 
    picture: gender, age, explain appearance (mention hair color, eye color), 
    the action they are doing for each person individually. 
    What is the body type the have slim/normal/a bit chubby.  Be as precise as 
    possible. Do not mention background. Do not mention clothes of person. 
    Use just 70 words. Do not mention my instructions. Mention what 
    hair would suit for each person in the style of 
"""

hair_prompt = ["""Bioluminescent odyssey""",
"""Luminescent waves""",
"""Obsidian waves""",
"""Platinum flux""",
]

clothes_prompt = ["""Clothes: Mission-driven pioneer ensembles in deep sapphire and violet gradients:  
tailored insulated jackets with articulated shoulder panels and hidden utility pockets,  
form-fitting hex-woven undersuits with moisture-wicking properties,  
reinforced elbow and knee patches echoing constellations in muted metallic thread,  
low-profile expedition boots with luminescent tread strips,  
and compact harness belts holding glowing data-glyph canisters.  
Subtle scuff marks and faint iridescent tracer lines along seams speak of trials past,  
while integrated fiber-optic filaments pulse like a calm heartbeat beneath the fabric.
""",
"""Clothes: Dust-worn workshop ensembles in layered navy and slate grey:  
soft-lined hoodies with reinforced elbow patches and hidden zippered tool pockets,  
rolled-up sleeves revealing hex-knit undershirts with moisture-wicking channels,  
long, coat-like work vests with abrasion-resistant panels and clipped-in utility straps,  
sturdy leather workboots with scuffed toes and integrated magnet clasps,  
and slim canvas belts hung with miniature wrenches, wire spools, and calibration tags.  
Faint oil stains and frayed cuffs speak to long hours at the bench,  
while subtle phosphor threading along seams pulses with the spark of new ideas.
""",
"""Clothes: Darkly refined explorer regalia in navy, charcoal, and violet undertones:  
tailored high-collar frock coats cut from matte-woven fabric with subtle sheen,  
crisp charcoal trousers with hidden map-pocket panels,  
deep-violet satin lapel inlays stamped with constellatory sigils,  
structured undershirts of moisture-wicking mesh threaded with microLED filaments,  
polished black leather boots with antique brass buckles,  
and slim utility harnesses carrying brass-capped vials and leather-bound data-scrolls.  
Fine graphite scuffing and faint luminescent embroidery along cuffs evoke both age-old lore and tomorrow’s frontier.
""",
"""Clothes: Minimalist science–explorer hybrids in stark white and graphite tones:  
crisp, knee-length lab coats cut from lightweight, weather-resistant fabric,  
underlain by slim, hex-weave techshirts with hidden utility panels,  
sleek charcoal boots with reinforced toes and magnetic fastening straps,  
and low-profile utility belts threaded with data-slates and compact instrument pods.  
Subtle matte-black details—zippered cuffs, articulated seams, and micro-LED status strips—  
speak to precision engineering, while faint graphite scuffing hints at long hours in the field.
"""
]

background_prompt = ["""Background: Luminous descent corridor suffused with mist and particles of light:  
towering walls ripple in soft aqua and violet luminescence,  
as if carved from liquid starlight.  
Floating motes drift like bioluminescent plankton in still water,  
and a gentle haze blurs the boundary between solid and void.  
The tunnel stretches into inky darkness, its vanishing point alive with silent promise—  
a calm, cinematic threshold brimming with humble determination and boundless possibility.
""",
"""Background: Repurposed cathedral of invention bathed in cool blue glow:  
vaulted steel rafters arch above workbenches strewn with broken gears and circuit boards,  
pillars of reclaimed piping entwine with neat rows of lab instruments,  
arched windows of frosted glass cast prismatic halos on concrete floors,  
and drifting particles of white light hover like stardust in the beams.  
A cinematic fusion of garage grit, lab precision, and hallowed space—  
an atmosphere alive with humble ambition and the heart of progress.
""",
"""Background: Timeless fusion of cosmos, calculus, and classical architecture:  
an ethereal starfield bleeds into faded chalk equations swirling across vaulted pillars,  
weathered marble steps ascend into drifting lavender fog,  
arched bookshelves float like monoliths in soft white-blue radiance,  
and ghostly volumes hover amid motes of starlight.  
A gentle haze diffuses the light, revealing silhouettes that bridge ancient legacy and futuristic wonder—  
a mystical, cinematic ode to curiosity’s eternal flame.
""",
"""Background: Nocturnal lab haven bathed in soft white and violet-blue radiance:  
corrugated metal walls streaked with phosphorized algae glow behind scattered paper scraps,  
weathered control panels hum under pools of diffused light,  
and a towering data-stream wall ripples with flowing code in gentle luminescence.  
A single metal case casts upturned shafts of light, illuminating old calibrators and notebooks—  
an intimate, cinematic tableau of dedication and quiet pride in the midnight hour.
"""
]

replicate_model = "ideogram-ai/ideogram-v3-turbo"