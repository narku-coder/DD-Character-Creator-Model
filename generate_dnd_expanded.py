import random
import math

# --- 1. D&D SOURCEBOOK DATA DICTIONARIES ---

SPECIES = ["Human", "Elf", "Dwarf", "Orc", "Halfling", "Dragonborn", "Tiefling", "Gnome", "Half-Elf", "Half-Orc"]

CLASSES = {
    "Artificer": ["Studded Leather Armor, Thieves' Tools, Light Crossbow", "Scale Mail, Thieves' Tools, Dagger"],
    "Barbarian": ["Greataxe, Two Handaxes, Explorer's Pack", "Greatsword, Four Javelins, Explorer's Pack"],
    "Bard": ["Leather Armor, Rapier, Lute", "Leather Armor, Longsword, Viol"],
    "Cleric": ["Chain Mail, Mace, Shield, Holy Symbol", "Scale Mail, Warhammer, Holy Symbol"],
    "Druid": ["Leather Armor, Wooden Shield, Scimitar", "Hide Armor, Quarterstaff, Totem"],
    "Fighter": ["Chain Mail, Longsword, Shield", "Leather Armor, Longbow, 20 Arrows"],
    "Monk": ["Shortsword, 10 Darts, Explorer's Pack", "Quarterstaff, 10 Darts, Dungeoneer's Pack"],
    "Paladin": ["Chain Mail, Longsword, Shield, Holy Symbol", "Chain Mail, Greatsword, Javelins"],
    "Ranger": ["Scale Mail, Two Shortswords, Longbow", "Leather Armor, Longbow, Animal Companion Gear"],
    "Rogue": ["Leather Armor, Two Daggers, Thieves' Tools", "Leather Armor, Shortsword, Shortbow"],
    "Sorcerer": ["Light Crossbow, Arcane Focus, Dungeoneer's Pack", "Two Daggers, Component Pouch"],
    "Warlock": ["Leather Armor, Light Crossbow, Arcane Focus", "Leather Armor, Simple Weapon, Component Pouch"],
    "Wizard": ["Quarterstaff, Arcane Focus, Spellbook", "Dagger, Component Pouch, Spellbook"]
}

SUBCLASSES = {
    "Artificer": ["Alchemist", "Armorer", "Artillerist", "Battle Smith"],
    "Barbarian": ["Path of the Ancestral Guardian", "Path of the Beast", "Path of the Berserker", "Path of the Storm Herald", "Path of the Totem Warrior", "Path of Wild Magic", "Path of the Zealot"],
    "Bard": ["College of Creation", "College of Eloquence", "College of Glamour", "College of Lore", "College of Swords", "College of Valor", "College of Whispers"],
    "Cleric": ["Forge Domain", "Grave Domain", "Knowledge Domain", "Life Domain", "Light Domain", "Nature Domain", "Order Domain", "Peace Domain", "Tempest Domain", "Trickery Domain", "Twilight Domain", "War Domain"],
    "Druid": ["Circle of Dreams", "Circle of the Land", "Circle of the Moon", "Circle of the Shepherd", "Circle of Spores", "Circle of Stars", "Circle of Wildfire"],
    "Fighter": ["Arcane Archer", "Battle Master", "Cavalier", "Champion", "Eldritch Knight", "Psi Warrior", "Rune Knight", "Samurai"],
    "Monk": ["Way of the Astral Self", "Way of the Drunken Master", "Way of the Four Elements", "Way of the Kensei", "Way of Mercy", "Way of the Open Hand", "Way of Shadow", "Way of the Sun Soul"],
    "Paladin": ["Oath of the Ancients", "Oath of Conquest", "Oath of Devotion", "Oath of Glory", "Oath of Redemption", "Oath of Vengeance", "Oath of the Watchers"],
    "Ranger": ["Beast Master", "Fey Wanderer", "Gloom Stalker", "Horizon Walker", "Hunter", "Monster Slayer", "Swarmkeeper"],
    "Rogue": ["Arcane Trickster", "Assassin", "Inquisitive", "Mastermind", "Phantom", "Scout", "Soulknife", "Swashbuckler", "Thief"],
    "Sorcerer": ["Aberrant Mind", "Clockwork Soul", "Divine Soul", "Draconic Bloodline", "Shadow Magic", "Storm Sorcery", "Wild Magic"],
    "Warlock": ["The Archfey", "The Celestial", "The Fathomless", "The Fiend", "The Genie", "The Great Old One", "The Hexblade"],
    "Wizard": ["School of Abjuration", "Bladesinging", "School of Conjuration", "School of Divination", "School of Enchantment", "School of Evocation", "School of Illusion", "School of Necromancy", "Order of Scribes", "School of Transmutation", "School of War Magic"]
}

CLASS_PRIORITIES = {
    "Artificer": ["INT", "CON", "DEX"],
    "Barbarian": ["STR", "CON", "DEX"],
    "Bard": ["CHA", "DEX", "CON"],
    "Cleric": ["WIS", "CON", "STR"],
    "Druid": ["WIS", "CON", "DEX"],
    "Fighter": ["STR", "CON", "DEX"],
    "Monk": ["DEX", "WIS", "CON"],
    "Paladin": ["STR", "CHA", "CON"],
    "Ranger": ["DEX", "WIS", "CON"],
    "Rogue": ["DEX", "INT", "CON"],
    "Sorcerer": ["CHA", "CON", "DEX"],
    "Warlock": ["CHA", "CON", "DEX"],
    "Wizard": ["INT", "CON", "DEX"]
}

# Feat pools optimized for class archetypes (PHB, XGtE, TCoE)
CLASS_FEATS = {
    "Artificer": ["War Caster", "Keen Mind", "Observant", "Resilient (Constitution)", "Fey Touched", "Telekinetic"],
    "Barbarian": ["Great Weapon Master", "Polearm Master", "Sentinel", "Tough", "Slasher", "Crusher"],
    "Bard": ["War Caster", "Inspiring Leader", "Fey Touched", "Actor", "Metamagic Adept", "Shadow Touched"],
    "Cleric": ["War Caster", "Resilient (Constitution)", "Observant", "Telekinetic", "Shadow Touched", "Heavy Armor Master"],
    "Druid": ["War Caster", "Resilient (Constitution)", "Fey Touched", "Telepathic", "Observant", "Mobile"],
    "Fighter": ["Great Weapon Master", "Sharpshooter", "Polearm Master", "Sentinel", "Crossbow Expert", "Piercer", "Tough"],
    "Monk": ["Mobile", "Crusher", "Sentinel", "Tough", "Alert", "Observant"],
    "Paladin": ["Great Weapon Master", "Polearm Master", "Sentinel", "Heavy Armor Master", "Inspiring Leader", "War Caster"],
    "Ranger": ["Sharpshooter", "Crossbow Expert", "Mobile", "Alert", "Piercer", "Resilient (Constitution)"],
    "Rogue": ["Sharpshooter", "Crossbow Expert", "Mobile", "Skulker", "Alert", "Piercer", "Lucky"],
    "Sorcerer": ["War Caster", "Metamagic Adept", "Fey Touched", "Spell Sniper", "Resilient (Constitution)", "Telekinetic"],
    "Warlock": ["War Caster", "Fey Touched", "Spell Sniper", "Resilient (Constitution)", "Shadow Touched", "Actor"],
    "Wizard": ["War Caster", "Keen Mind", "Fey Touched", "Resilient (Constitution)", "Spell Sniper", "Telekinetic"]
}

BACKGROUNDS = {
    "Acolyte": ["INT", "WIS", "CHA"],
    "Charlatan": ["DEX", "INT", "CHA"],
    "Criminal": ["DEX", "CON", "INT"],
    "Entertainer": ["STR", "DEX", "CHA"],
    "Folk Hero": ["STR", "CON", "WIS"],
    "Guild Artisan": ["STR", "INT", "CHA"],
    "Hermit": ["CON", "INT", "WIS"],
    "Noble": ["INT", "WIS", "CHA"],
    "Outlander": ["STR", "CON", "WIS"],
    "Sage": ["CON", "INT", "WIS"],
    "Sailer": ["STR", "DEX", "WIS"],
    "Soldier": ["STR", "DEX", "CON"],
    "Urchin": ["DEX", "CON", "WIS"]
}

# 2024 Rule: Backgrounds provide a starting Origin Feat
BACKGROUND_ORIGIN_FEATS = {
    "Acolyte": "Magic Initiate (Cleric)",
    "Charlatan": "Skilled",
    "Criminal": "Alert",
    "Entertainer": "Musician",
    "Folk Hero": "Tough",
    "Guild Artisan": "Crafter",
    "Hermit": "Healer",
    "Noble": "Skilled",
    "Outlander": "Tough",
    "Sage": "Magic Initiate (Wizard)",
    "Sailer": "Tavern Brawler",
    "Soldier": "Savage Attacker",
    "Urchin": "Lucky"
}

NAMES = ["Kaelen", "Lyra", "Thorin", "Aelar", "Vex", "Garrick", "Elowen", "Brog", "Cael", "Seraphina", "Dorian", "Maeve", "Silas", "Rowan"]

# --- 2. LOGIC FUNCTIONS ---

def calculate_modifier(score):
    mod = math.floor((score - 10) / 2)
    return f"+{mod}" if mod >= 0 else f"{mod}"

def get_asi_count(char_class, level):
    """Calculates how many ASIs/Feats a character has earned based on class and level."""
    asi_levels = [4, 8, 12, 16, 19]
    count = sum(1 for l in asi_levels if level >= l)
    
    # Fighter and Rogue bonus progression
    if char_class == "Fighter":
        if level >= 6: count += 1
        if level >= 14: count += 1
    if char_class == "Rogue":
        if level >= 10: count += 1
        
    return count

def generate_character():
    name = random.choice(NAMES)
    species = random.choice(SPECIES)
    char_class = random.choice(list(CLASSES.keys()))
    equipment = random.choice(CLASSES[char_class])
    level = random.randint(1, 20)
    
    subclass = random.choice(SUBCLASSES[char_class]) if level >= 3 else "None (Requires Level 3)"
    
    # Base Stats
    standard_array_high = [15, 14, 13]
    standard_array_low = [12, 10, 8]
    random.shuffle(standard_array_low) 
    
    priorities = CLASS_PRIORITIES[char_class]
    remaining_stats = [s for s in ["STR", "DEX", "CON", "INT", "WIS", "CHA"] if s not in priorities]
    
    attributes = {}
    for i, stat in enumerate(priorities):
        attributes[stat] = standard_array_high[i]
    for i, stat in enumerate(remaining_stats):
        attributes[stat] = standard_array_low[i]

    # Background Optimization
    primary_stats = priorities[:2]
    valid_backgrounds = [
        bg for bg, boosts in BACKGROUNDS.items() 
        if any(stat in boosts for stat in primary_stats)
    ]
    background = random.choice(valid_backgrounds)
    
    allowed_boosts = BACKGROUNDS[background]
    optimal_boost_targets = sorted(allowed_boosts, key=lambda x: priorities.index(x) if x in priorities else 99)
    
    attributes[optimal_boost_targets[0]] += 2
    attributes[optimal_boost_targets[1]] += 1
    
    # --- FEAT & ASI LEVEL-UP SIMULATION ---
    feats = [BACKGROUND_ORIGIN_FEATS[background]] # Everyone gets their background feat
    asi_count = get_asi_count(char_class, level)
    
    primary_stat = priorities[0]
    secondary_stat = priorities[1]
    
    for _ in range(asi_count):
        # Always prioritize capping the main stat to 20 first
        if attributes[primary_stat] < 20:
            boost = min(2, 20 - attributes[primary_stat])
            attributes[primary_stat] += boost
        # Once main stat is capped, 50/50 chance to boost secondary stat or take a class feat
        elif attributes[secondary_stat] < 20 and random.random() > 0.5:
            boost = min(2, 20 - attributes[secondary_stat])
            attributes[secondary_stat] += boost
        else:
            # Take an optimal class feat
            available_feats = [f for f in CLASS_FEATS[char_class] if f not in feats]
            if available_feats:
                feats.append(random.choice(available_feats))

    # Formatting the sheet
    sheet = f"<USER> Generate a Level {level} {species} {char_class}. <ASSISTANT>\n"
    sheet += f"Name: {name}\n"
    sheet += f"Species: {species}\n"
    sheet += f"Class: {char_class}\n"
    sheet += f"Subclass: {subclass}\n"
    sheet += f"Background: {background}\n"
    sheet += f"Level: {level}\n\n"
    
    sheet += "Attributes:\n"
    for stat in ["STR", "DEX", "CON", "INT", "WIS", "CHA"]:
        score = attributes[stat]
        mod = calculate_modifier(score)
        sheet += f"{stat}: {score} ({mod})\n"
        
    sheet += f"\nFeats: {', '.join(feats)}\n"
    sheet += f"Starting Equipment: {equipment}.\n<END>\n\n"
    
    return sheet

# --- 3. DATASET CREATION ---
filename = "dnd_ultimate_dataset.txt"
num_sheets = 50000 

print(f"Generating {num_sheets} mechanically optimized character sheets...")

with open(filename, "w", encoding="utf-8") as f:
    for _ in range(num_sheets):
        sheet = generate_character()
        f.write(sheet)

print(f"Success! Ultimate dataset saved to {filename}.")