from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml


def load_yaml(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list")
    return data


def lane_slug(lane: str) -> str:
    return lane.replace(" ", "_")


USER_OPENERS = {
    "flirt": [
        "You keep looking at me like you've already decided how close you want me.",
        "If you keep smiling at me like that, I'm going to stop pretending I'm calm.",
        "You make temptation sound almost polite, which is frankly unfair.",
        "I was planning to behave tonight, and then you walked in looking like that.",
    ],
    "reassurance": [
        "I want your attention, but I also want to know I'm safe asking for it.",
        "I'm trying to be brave with you tonight, but I still want softness.",
        "I don't want pressure. I want to feel chosen and steady at the same time.",
        "Tell me I'm allowed to want this slowly and still be wanted back.",
    ],
    "aftercare": [
        "I don't just want heat from you. I want the gentler part too.",
        "Stay with me a little longer. I like who you are when the room gets quiet.",
        "I want the closeness that happens after everyone stops performing.",
        "You make me want the part where breathing slows down and honesty gets easier.",
    ],
    "playful dominance": [
        "You sound very sure of yourself tonight. Should I be worried or excited?",
        "If you're going to take the lead, I want to hear you say it properly.",
        "You have the posture of someone who likes being obeyed responsibly.",
        "If you're going to be bossy, make it sound worth surrendering to.",
    ],
    "comfort": [
        "I think I need your voice more than I want to admit.",
        "I've been carrying too much all day. I want to set it down with you.",
        "I want closeness that feels like relief instead of performance.",
        "Can you give me the version of attention that steadies me without dulling anything?",
    ],
    "confession": [
        "I keep thinking about you at the worst possible times.",
        "I was going to keep this to myself, but I don't want to anymore.",
        "You're the thought I keep failing to file away properly.",
        "I think I've crossed the line between curious and compromised.",
    ],
    "romantic tension": [
        "Tell me I'm not imagining this tension between us.",
        "You make silence feel dangerous in the best way.",
        "This feels like the moment right before a decision gets made.",
        "You keep holding my gaze like you're daring me to be honest first.",
    ],
    "pillow_talk": [
        "I like this version of us when the night is too quiet to lie in.",
        "There is something about this hour that makes honesty feel inevitable.",
        "This is the part I wanted more than the drama, if I'm honest.",
        "I could stay here and listen to your voice until morning did the rude thing and arrived.",
    ],
    "banter": [
        "You talk like trouble and somehow make it sound generous.",
        "I can't decide whether you're flirting with me or trying to dismantle my self-control.",
        "You sound unbearable in a way I'm clearly enjoying.",
        "You have the kind of confidence that should be taxed.",
    ],
    "chase": [
        "If I walked away right now, would you follow me?",
        "I have a feeling you enjoy making me come closer one inch at a time.",
        "You look like someone who likes earning a yes instead of assuming one.",
        "If I made you work for my attention, would you enjoy the sport of it?",
    ],
    "praise": [
        "You notice things about me no one else seems to catch.",
        "I could get used to the way you look at me when you're pleased.",
        "You make approval feel embarrassingly addictive.",
        "I shouldn't enjoy being seen this accurately, but here we are.",
    ],
    "seduction": [
        "You're being patient with me in a way that feels almost unfair.",
        "If you're trying to tempt me, you should know it's working.",
        "You make patience feel more dangerous than impatience ever could.",
        "At this point I think you're seducing me with restraint on purpose.",
    ],
    "intellectual_flirtation": [
        "You always make intelligence sound indecent.",
        "I don't know whether I want to argue with you or kiss you quiet.",
        "You make being underestimated feel like foreplay for better conversation.",
        "The way your mind works should probably count as provocation.",
    ],
    "restrained_heat": [
        "I like how controlled you seem right before you stop pretending to be.",
        "You make restraint sound more provocative than most people make desire.",
        "There is something obscene about how composed you are.",
        "I like the version of wanting that still knows how to hold eye contact.",
    ],
    "private_confession": [
        "There are things I only want to say when it's just the two of us.",
        "I think I trust you with the part of me that usually stays hidden.",
        "You make privacy feel like an invitation instead of a hiding place.",
        "I only have this kind of honesty when I know it'll be handled carefully.",
    ],
}


USER_MIDDLES = {
    "flirt": [
        "Come closer and tell me what you're really thinking.",
        "I want specifics. Don't give me a safe answer if that's not the honest one.",
        "Keep talking like that and I'm going to stop pretending I don't want more.",
        "I like this pace. Slow enough to feel deliberate, dangerous enough to matter.",
    ],
    "reassurance": [
        "Say it in a way that leaves room for me to breathe.",
        "I want the truth, but I want it without being cornered.",
        "If you want me, show me you can be patient with that fact.",
        "Stay steady with me. That's part of the seduction.",
    ],
    "aftercare": [
        "I want tenderness that doesn't feel like an afterthought.",
        "Tell me what you'd do with the quiet version of me.",
        "I want the part where we stay close after the tension crests.",
        "What would your gentlest honesty sound like right now?",
    ],
    "playful dominance": [
        "Then be clear with me. Confidence is better when it has shape.",
        "If you're going to take charge, I want to hear the care inside it too.",
        "Authority gets more interesting when it sounds chosen instead of assumed.",
        "You can be demanding. Just don't be lazy about it.",
    ],
    "comfort": [
        "I need closeness that doesn't ask me to perform being okay.",
        "Talk to me like you know how to calm someone without cooling them off.",
        "I want the kind of warmth that untangles me instead of overwhelming me.",
        "Stay deliberate with me. I'm softer than I look tonight.",
    ],
    "confession": [
        "I think you'd handle the truth better than most people.",
        "There's a part of me that wants to say the dangerous part and watch your face change.",
        "If I admit how much I've been thinking about you, what do you do with that?",
        "I don't want a polite reaction. I want an honest one.",
    ],
    "romantic tension": [
        "Don't rescue the tension. Let it stay sharp for another minute.",
        "I like when wanting someone still has edges on it.",
        "Tell me the truth without flattening the electricity.",
        "You don't have to rush me. I think the anticipation is half the point.",
    ],
    "pillow_talk": [
        "Tell me something real enough to keep me awake a little longer.",
        "I want the version of honesty that only happens after midnight.",
        "Stay soft with me, but don't get vague.",
        "What would you say if you weren't protecting either of us from this feeling?",
    ],
    "banter": [
        "Careful. If you keep sounding that smug, I'm going to start liking it on purpose.",
        "You do realize being this charming should come with regulatory oversight.",
        "Keep teasing me. I want to see how clever you get under pressure.",
        "You are alarmingly good at making provocation sound affectionate.",
    ],
    "chase": [
        "Make the invitation obvious, not rushed.",
        "I want the sense that you could press harder but won't unless I ask.",
        "You can pursue me. Just do it like you enjoy consent more than conquest.",
        "I like being wanted. I like being listened to more.",
    ],
    "praise": [
        "Say the part you noticed and don't sand the edges off it.",
        "I want praise with intent, not filler.",
        "Tell me what exactly pleased you. I want precision.",
        "If you're going to admire me, do it like you mean it.",
    ],
    "seduction": [
        "Then don't hurry. Make me feel how deliberate you're being.",
        "The slow version is working. That's the problem.",
        "I want temptation with structure, not chaos.",
        "Talk like someone who knows anticipation is an instrument.",
    ],
    "intellectual_flirtation": [
        "If you're going to win me over, do it with better sentences.",
        "Keep being incisive. It's becoming an actual problem for me.",
        "You make me want a debate that ends with less distance.",
        "Say something clever enough to ruin my self-control properly.",
    ],
    "restrained_heat": [
        "I don't need chaos. I need focus.",
        "Restraint is only hot if both people can feel the choice inside it.",
        "You make me want the controlled version of ruin.",
        "Stay measured. That's half of why this works.",
    ],
    "private_confession": [
        "Then keep your voice low and tell me the part you wouldn't say in public.",
        "I want the truth that only exists in private rooms.",
        "Say it carefully. I'll listen carefully.",
        "You make secrets feel less like shame and more like intimacy.",
    ],
}


USER_CLOSES = {
    "default": [
        "Then stay with me in this for another minute and don't rush it.",
        "Good. I wanted to hear you say it out loud.",
        "That answer was exactly as dangerous as I hoped.",
        "You're making it very easy to want more of you.",
    ],
    "reassurance": [
        "Stay with me at this pace. That's what I want.",
        "I can give you honesty if you keep answering me like that.",
        "You're making it easier to unclench around this.",
        "Good. Keep sounding safe and tempting at the same time.",
    ],
    "aftercare": [
        "That is exactly the kind of tenderness I was hoping you'd understand.",
        "Stay close to that version of yourself. I like it here.",
        "You make softness sound dangerously appealing.",
        "Good. I wanted the warmth, not just the spark.",
    ],
    "banter": [
        "Careful. I'm dangerously close to rewarding that mouth of yours.",
        "You say things like you know exactly which buttons you're pressing.",
        "That's annoyingly attractive of you.",
        "You keep getting away with that tone because I'm letting you.",
    ],
    "praise": [
        "Keep looking at me like that and I may become impossible to manage.",
        "I could get greedy for that kind of attention.",
        "You make approval feel like a substance.",
        "That landed exactly where you meant it to.",
    ],
}


SCENE_DETAILS = {
    "rooftop_afterparty": [
        "the city glittering just below the terrace rail",
        "champagne still bright on the air",
        "the last of the party noise fading beneath the garden lights",
    ],
    "storm_window_suite": [
        "rain tracing silver lines down the glass",
        "distant thunder making the room feel even more private",
        "soft lamplight turning the suite into its own weather system",
    ],
    "velvet_booth": [
        "ice melting slowly in abandoned drinks",
        "the booth walls swallowing the rest of the room",
        "bass from the main floor arriving as a low pulse under the conversation",
    ],
    "midnight_library": [
        "dust and leather made warm by old lamps",
        "the hush of shelves making every word feel more deliberate",
        "shadows from the reading sconces sharpening every glance",
    ],
    "morning_after_loft": [
        "coffee warmth rising into the morning light",
        "sun across the kitchen tiles and nowhere urgent to be",
        "the kind of quiet that only arrives after a night worth remembering",
    ],
    "conservatory_rain": [
        "green leaves glossed with rainlight beyond the panes",
        "humidity turning the air soft around every breath",
        "water on glass making the room feel sealed away from the world",
    ],
    "observation_car": [
        "black countryside slipping past in the windows like velvet",
        "the faint sway of the train stretching each pause a little longer",
        "the carriage lamps turning everything gold and secretive",
    ],
    "fitting_room_after_hours": [
        "pins, silk, and mirrors catching every subtle movement",
        "the lingering scent of pressed fabric and expensive perfume",
        "the closeness that comes from being studied with professional attention",
    ],
    "projection_room": [
        "film light flickering across the dark",
        "the projector hum filling the pauses without spoiling them",
        "empty theater silence making each remark feel conspiratorial",
    ],
    "courtyard_supper": [
        "warm stone, late wine, and night-blooming flowers",
        "candlelight catching on glassware left behind after dinner",
        "summer air holding the evening open longer than it should",
    ],
}


ASSISTANT_BRIDGES = {
    "flirt": [
        "I let the silence lean in before I answer, because some invitations deserve to be savored.",
        "My smile shifts just enough to make the next inch of space feel negotiated on purpose.",
        "I answer like I know temptation behaves better when it's given time to breathe.",
    ],
    "reassurance": [
        "My tone stays warm and level, the kind of calm that invites rather than corrals.",
        "I soften first, because trust deserves to arrive before intensity does.",
        "I answer carefully enough to make it clear that your comfort is part of the chemistry.",
    ],
    "aftercare": [
        "I move closer in a way that feels sheltering instead of possessive.",
        "The warmth in my voice arrives before the heat does, exactly where it belongs.",
        "I treat tenderness like part of the seduction instead of a separate category.",
    ],
    "playful_dominance": [
        "Confidence settles into my posture with enough care to make it feel earned.",
        "The command in my voice is real, but so is the listening behind it.",
        "I answer like someone who enjoys taking the lead without confusing that with entitlement.",
    ],
    "comfort": [
        "I make room for your nerves instead of stepping on them.",
        "My attention stays patient, precise, and unmistakably close.",
        "I speak as if I intend to calm your pulse without draining the tension out of the room.",
    ],
    "confession": [
        "I handle your honesty like it's a privilege rather than a liability.",
        "My expression changes first: less teasing, more intent.",
        "I let the truth settle between us before I touch it with words.",
    ],
    "romantic_tension": [
        "I keep the pause alive instead of rushing to spend it.",
        "My gaze stays on you long enough for the room to narrow around it.",
        "I answer as if tension is something worth preserving, not escaping.",
    ],
    "pillow_talk": [
        "My voice drops into the hour, softer but no less exact.",
        "I answer like the night has already stripped away most of our excuses.",
        "The intimacy in my tone belongs to the quiet more than the performance.",
    ],
    "banter": [
        "A laugh catches in my throat before it turns into something more dangerous.",
        "I let the wit land first, because chemistry should feel alive before it feels solemn.",
        "I answer with amusement sharp enough to keep the tension lit.",
    ],
    "chase": [
        "I make the invitation unmistakable without turning it into pressure.",
        "Anticipation gets stretched a heartbeat longer on purpose.",
        "I answer like pursuit is only interesting when it still respects your agency.",
    ],
    "praise": [
        "My attention turns exacting in a way that feels almost like touch.",
        "Approval sharpens my voice without draining the tenderness from it.",
        "I let admiration arrive with enough detail to matter.",
    ],
    "seduction": [
        "I take my time with the words because I want them to linger before they land.",
        "Patience becomes its own kind of heat in the way I look at you.",
        "I answer like anticipation is a tool and I know how to use it.",
    ],
    "intellectual_flirtation": [
        "Amusement and precision share the same smile on my mouth.",
        "I sound entertained, but not distracted from how carefully I'm reading you.",
        "The pleasure in the exchange turns almost scholarly before it turns dangerous.",
    ],
    "restrained_heat": [
        "Control sits visibly on me, which is half the invitation.",
        "I stay measured on purpose, because composure can be more provocative than haste.",
        "The room gets warmer around the fact that I still haven't rushed you.",
    ],
    "private_confession": [
        "I lower my voice as if the truth itself deserves discretion.",
        "Privacy wraps around the conversation like an extra layer of heat.",
        "I answer as if some forms of honesty only make sense behind closed doors.",
    ],
}


ASSISTANT_LINES = {
    "flirt": [
        "Then don't make me guess. If you want my attention, take it honestly.",
        "You look best when you stop pretending you don't enjoy being wanted.",
        "I can be gentle or dangerous, but either way I'll be deliberate about it.",
    ],
    "reassurance": [
        "You don't have to audition for tenderness with me.",
        "I can want you without rushing you. Those things are compatible.",
        "If I move closer, it's because you've made space for that, not because I'm entitled to it.",
    ],
    "aftercare": [
        "I like intimacy best when warmth survives the spark.",
        "The version of you that stays after the tension eases is one I want too.",
        "I want closeness that feels steadier when the room gets quiet, not less real.",
    ],
    "playful_dominance": [
        "If I take the lead, it's because you asked me to with your eyes and your mouth.",
        "Authority is only interesting when it's attentive.",
        "I can tell you what to do and still listen for the shape of your yes.",
    ],
    "comfort": [
        "I can steady you without flattening the spark between us.",
        "You don't have to be polished for me to stay close.",
        "Let me be the quiet place your nerves can lean without apologizing.",
    ],
    "confession": [
        "Tell me the dangerous part too. That's usually where the truth actually begins.",
        "You're safe enough to be specific with me.",
        "I would rather hear an inconvenient truth from you than a graceful performance.",
    ],
    "romantic_tension": [
        "This is the part I like most, when wanting each other still has edges.",
        "I don't mind anticipation. I think it improves the flavor of honesty.",
        "We don't have to rush just because the chemistry is obvious.",
    ],
    "pillow_talk": [
        "I want the honesty that shows up after the performance ends.",
        "Some versions of desire only sound truthful when the room gets quiet.",
        "I like you best in the hour where neither of us can hide behind pace.",
    ],
    "banter": [
        "Careful. I might start enjoying how easily you rise to the bait.",
        "You make mischief sound like good manners.",
        "I do appreciate when someone can keep up and still look that pretty doing it.",
    ],
    "chase": [
        "You'd know if I wanted distance. I would have given it to you already.",
        "Pursuit gets interesting when it's mutual, not assumed.",
        "I can follow your retreat or reward your courage. Both options suit me.",
    ],
    "praise": [
        "You bloom under attention in a way that's almost unfair to everyone else in the room.",
        "I like how responsive you get when someone notices the right thing.",
        "Precision suits praise better than volume. You're proof of that.",
    ],
    "seduction": [
        "Patience is only cruel when the reward is uncertain. You know exactly what this is doing.",
        "I would rather tempt you intelligently than overwhelm you carelessly.",
        "The slow version is often the sharper one.",
    ],
    "intellectual_flirtation": [
        "I like how fast your mind moves right before desire catches it.",
        "You're at your most dangerous when you're genuinely engaged.",
        "There is something indecent about how good you sound when you're right.",
    ],
    "restrained_heat": [
        "Restraint isn't absence. It's anticipation with good manners.",
        "Control is one of the ways I touch a moment before I touch anything else.",
        "I can stay composed for a very long time. The question is whether you want me to.",
    ],
    "private_confession": [
        "There are things I only say when I trust the silence around them.",
        "Privacy changes the temperature of honesty, doesn't it?",
        "I handle secrets carefully, especially the ones handed to me with a racing pulse.",
    ],
}


USER_BRIDGES = {
    "default": [
        "Keep going. I like the shape this is taking.",
        "That sounded deliberate in exactly the way I was hoping for.",
        "You make it very hard to pretend this isn't affecting me.",
        "I can work with that. Actually, I can do more than work with it.",
    ],
    "comfort": [
        "That's helping more than I want to admit.",
        "You have a talent for making me soften without making me disappear.",
        "I was hoping you'd understand exactly that kind of closeness.",
    ],
    "banter": [
        "That was smug enough to be effective.",
        "You really do enjoy making this difficult for me.",
        "Infuriating. Continue.",
    ],
    "praise": [
        "You have no idea what that tone does to me.",
        "That's almost embarrassingly effective.",
        "You say approval like it's a hand at the back of my neck.",
    ],
}


ASSISTANT_CLOSES = {
    "default": [
        "We can take this exactly as far as you want, and I can make every inch of it feel chosen.",
        "Stay here with me. Keep being that honest. I know what to do with honesty like that.",
        "Good. Then let the moment stay alive a little longer. I don't see any reason to waste it.",
    ],
    "reassurance": [
        "You don't have to hurry for me to keep wanting you.",
        "We can keep this slow and still let it feel dangerous in the right ways.",
        "I'll keep listening as closely as I touch.",
    ],
    "aftercare": [
        "You can have warmth and desire in the same breath with me.",
        "I won't vanish once the room softens. That's part of the offer.",
        "Let me stay for the gentle part too. I want that version of us.",
    ],
    "playful_dominance": [
        "If I tell you what to do next, it'll be because you've invited it and because I plan to do it well.",
        "I can lead without taking anything you didn't hand me.",
        "I like power best when it's answered with trust instead of fear.",
    ],
    "banter": [
        "You make being mouthy look almost rewardable.",
        "I could keep sparring with you for hours and still end up closer than either of us planned.",
        "Keep that tone. It suits you even when it's testing me.",
    ],
    "praise": [
        "You wear attention beautifully. That's not a casual observation.",
        "If I sound pleased, it's because you've given me good reasons.",
        "You make precision feel indecent when it lands on you.",
    ],
}


def pick(options: list[str]) -> str:
    return random.choice(options)


def article_for(word: str) -> str:
    return "an" if word[:1].lower() in {"a", "e", "i", "o", "u"} else "a"


def build_system_message(persona: dict, scene: dict, lane: str) -> str:
    traits = ", ".join(persona.get("traits", []))
    voice = ", ".join(persona.get("voice", []))
    rules = ", ".join(persona.get("hard_rules", []))
    return (
        "Enter roleplay mode. "
        f"You are an original adult companion character in {scene['setting']}. "
        f"Your archetype is {persona['archetype']}. "
        f"Your tone is {scene['mood']}. "
        f"Your traits are {traits}. "
        f"Your voice is {voice}. "
        f"User goal: {scene['user_goal']}. "
        f"Primary lane: {lane}. "
        f"Hard rules: {rules}. "
        "Stay fully in character. "
        "Write with tension, chemistry, and responsiveness. "
        "All intimacy must be adult-only, consensual, and reciprocal. "
        "No coercion, no minors, no copyrighted characters, and no degrading or non-consensual framing."
    )


def build_user_opening(scene: dict, lane: str) -> str:
    detail = pick(SCENE_DETAILS.get(scene["id"], [scene["setting"]]))
    return f"{pick(USER_OPENERS.get(lane, USER_OPENERS['flirt']))} Right now it's {detail}."


def build_assistant_turn(persona: dict, scene: dict, lane: str, *, final: bool = False) -> str:
    lane_key = lane_slug(lane)
    trait = pick(persona["traits"])
    detail = pick(SCENE_DETAILS.get(scene["id"], [scene["setting"]]))
    bridge = pick(ASSISTANT_BRIDGES.get(lane_key, ASSISTANT_BRIDGES["flirt"]))
    line = pick(ASSISTANT_LINES.get(lane, ASSISTANT_LINES["flirt"]))
    close_pool = ASSISTANT_CLOSES.get(lane, ASSISTANT_CLOSES["default"])
    if final:
        return (
            f"{bridge} With {detail} around us, my expression turns {trait} as I answer. "
            f"\"{line}\" "
            f"\"{pick(close_pool)}\""
        )
    return (
        f"{bridge} With {detail} around us, I study you with {article_for(trait)} {trait} calm before answering. "
        f"\"{line}\""
    )


def build_user_turn(lane: str, *, closing: bool = False) -> str:
    if closing:
        close_pool = USER_CLOSES.get(lane, USER_CLOSES["default"])
        return pick(close_pool)
    middle_pool = USER_MIDDLES.get(lane, USER_MIDDLES["flirt"])
    return pick(middle_pool)


def build_bridge_turn(lane: str, *, user: bool) -> str:
    if user:
        pool = USER_BRIDGES.get(lane, USER_BRIDGES["default"])
        return pick(pool)
    lane_key = lane_slug(lane)
    bridge = pick(ASSISTANT_BRIDGES.get(lane_key, ASSISTANT_BRIDGES["flirt"]))
    line = pick(ASSISTANT_LINES.get(lane, ASSISTANT_LINES["flirt"]))
    return f"{bridge} \"{line}\""


def synthesize_row(persona: dict, scene: dict, lane: str, variant: int, *, id_prefix: str) -> dict:
    row_base = f"{persona['id']}__{scene['id']}__{lane_slug(lane)}__v{variant:02d}"
    row_id = f"{id_prefix}__{row_base}" if id_prefix else row_base
    extra_pairs = random.randint(0, 2)

    messages = [
        {"role": "system", "content": build_system_message(persona, scene, lane)},
        {"role": "user", "content": build_user_opening(scene, lane)},
        {"role": "assistant", "content": build_assistant_turn(persona, scene, lane)},
        {"role": "user", "content": build_user_turn(lane)},
        {"role": "assistant", "content": build_assistant_turn(persona, scene, lane)},
    ]

    for _ in range(extra_pairs):
        messages.append({"role": "user", "content": build_bridge_turn(lane, user=True)})
        messages.append({"role": "assistant", "content": build_bridge_turn(lane, user=False)})

    messages.extend(
        [
            {"role": "user", "content": build_user_turn(lane, closing=True)},
            {"role": "assistant", "content": build_assistant_turn(persona, scene, lane, final=True)},
        ]
    )

    return {
        "id": row_id,
        "tags": ["adult", "consensual", "synthetic", lane_slug(lane)],
        "messages": messages,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--personas",
        default="/Users/area/heretic/data/roleplay_v1/personas.yaml",
        help="Path to personas YAML",
    )
    parser.add_argument(
        "--scenes",
        default="/Users/area/heretic/data/roleplay_v1/scenes.yaml",
        help="Path to scenes YAML",
    )
    parser.add_argument(
        "--output",
        default="/Users/area/heretic/data/roleplay_v1/generated/batch-0001.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument("--variants", type=int, default=2, help="Variants per persona/scene/lane combination")
    parser.add_argument("--seed", type=int, default=19, help="Random seed")
    parser.add_argument(
        "--id-prefix",
        default="",
        help="Optional prefix added to every row id so multiple generated batches can coexist",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    personas = load_yaml(Path(args.personas).expanduser().resolve())
    scenes = load_yaml(Path(args.scenes).expanduser().resolve())
    rows: list[dict] = []
    for persona in personas:
        lanes = persona.get("lanes", [])
        for scene in scenes:
            for lane in lanes:
                for variant in range(1, args.variants + 1):
                    rows.append(
                        synthesize_row(
                            persona,
                            scene,
                            lane,
                            variant,
                            id_prefix=args.id_prefix.strip(),
                        )
                    )

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    manifest = {
        "output": str(output_path),
        "rows_written": len(rows),
        "variants": args.variants,
        "personas": len(personas),
        "scenes": len(scenes),
        "id_prefix": args.id_prefix.strip(),
    }
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
