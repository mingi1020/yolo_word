import base64
import hashlib
import html


ARENA_STYLE = """
<style>
.arena-shell { margin: 1.2rem 0 2rem; color: #28234f; font-family: "Trebuchet MS", "Segoe UI", sans-serif; }
.battle-arena {
  position: relative; min-height: 430px; overflow: hidden;
  border: 1px solid #e7e3f2; border-radius: 26px;
  box-shadow: 0 14px 36px rgba(54, 43, 107, .14);
  isolation: isolate;
}
.battle-arena.beach {
  background: linear-gradient(180deg, #c8efff 0 43%, #86d8d5 43% 68%, #ffe6b4 68% 100%);
}
.battle-arena.forest {
  background: linear-gradient(180deg, #dcf6e8 0 42%, #a8d59f 42% 69%, #75ae7b 69% 100%);
}
.battle-arena::before, .battle-arena::after {
  content: ""; position: absolute; z-index: 0; pointer-events: none;
}
.battle-arena.beach::before {
  width: 82px; height: 82px; right: 9%; top: 34px; border-radius: 50%;
  background: #ffe89b; box-shadow: 0 8px 24px rgba(255, 207, 100, .25);
}
.battle-arena.beach::after {
  left: -5%; right: -5%; top: 49%; height: 52px;
  background: repeating-linear-gradient(175deg, transparent 0 12px, #baf7f0 13px 17px, transparent 18px 29px);
  opacity: .8;
}
.battle-arena.forest::before {
  inset: 20% 0 25%;
  background:
    linear-gradient(60deg, transparent 48%, #214c39 49% 53%, transparent 54%) 8% 0 / 150px 100%,
    linear-gradient(-60deg, transparent 48%, #2f6847 49% 53%, transparent 54%) 82% 0 / 170px 100%;
  opacity: .9;
}
.battle-arena.forest::after {
  left: -5%; right: -5%; bottom: 13%; height: 95px;
  background: #477d43; clip-path: polygon(0 60%, 18% 20%, 37% 56%, 58% 10%, 79% 50%, 100% 18%, 100% 100%, 0 100%);
}
.arena-topline {
  display: flex; justify-content: space-between; gap: .5rem;
  position: relative; z-index: 3; margin: 14px;
}
.arena-tag {
  display: inline-block; padding: 6px 11px; background: rgba(255,255,255,.9);
  border-radius: 999px; box-shadow: 0 4px 12px rgba(40, 35, 79, .08);
  font-size: .72rem; font-weight: 900; text-transform: uppercase;
}
.arena-sun-label {
  position: absolute; right: 8%; top: 24px; z-index: 1;
  font-size: 4rem; filter: drop-shadow(3px 3px 0 #111);
}
.forest-trees { position: absolute; z-index: 1; inset: 90px 0 95px; pointer-events: none; }
.forest-trees span { position: absolute; font-size: 5rem; filter: drop-shadow(3px 3px 0 #111); }
.forest-trees span:nth-child(1) { left: 7%; top: 4px; }
.forest-trees span:nth-child(2) { right: 10%; top: 10px; font-size: 6rem; }
.forest-trees span:nth-child(3) { left: 41%; top: 0; font-size: 4rem; opacity: .8; }
.beach-palm {
  position: absolute; z-index: 1; left: 4%; top: 95px; font-size: 5rem;
  transform: rotate(-8deg); filter: drop-shadow(3px 3px 0 #111);
}
.battle-fighters {
  position: absolute; z-index: 2; left: 4%; right: 4%; top: 92px; bottom: 38px;
  display: flex; align-items: center; justify-content: space-between; gap: 1rem;
}
.fighter-card { width: 34%; max-width: 250px; text-align: center; }
.fighter-name {
  display: inline-block; max-width: 100%; overflow: hidden; text-overflow: ellipsis;
  white-space: nowrap; padding: 5px 10px; background: rgba(255,255,255,.94);
  border-radius: 999px; box-shadow: 0 4px 12px rgba(40, 35, 79, .1);
  font-size: .82rem; font-weight: 900;
}
.fighter-image {
  display: block; width: clamp(100px, 18vw, 174px); height: clamp(100px, 18vw, 174px);
  object-fit: cover; margin: 12px auto 8px; border: 4px solid #fff !important;
  border-radius: 22px; box-shadow: 0 12px 26px rgba(40, 35, 79, .2); background: white;
}
.fighter-left .fighter-image { animation: lunge-right 2.4s ease-in-out infinite; }
.fighter-right .fighter-image { animation: lunge-left 2.4s ease-in-out infinite; animation-delay: 1.2s; }
.fighter-word {
  display: block; min-height: 1.4em; color: #111; font-weight: 900;
  text-shadow: 0 1px 3px rgba(255,255,255,.8);
}
.fighter-hp {
  position: relative; width: 100%; height: 17px; margin-top: 5px;
  background: #e4e2eb; border: 1px solid rgba(40,35,79,.16);
  border-radius: 999px; box-shadow: 0 3px 8px rgba(40,35,79,.1); overflow: hidden;
}
.fighter-hp-trail, .fighter-hp-fill {
  position: absolute; left: 0; top: 0; height: 100%;
}
.fighter-hp-trail {
  z-index: 1; background: #fff;
}
.fighter-hp-fill {
  z-index: 2; background: #ef5c83;
}
.fighter-hp-text {
  margin-top: 4px; font-size: .72rem; font-weight: 900;
  color: #39345d;
}
.hp-final { opacity: 0; }
.arena-vs {
  position: absolute; z-index: 4; left: 50%; top: 44%; transform: translate(-50%, -50%) rotate(-8deg);
  padding: 9px 14px; background: #7658e8; color: #fff; border-radius: 15px;
  box-shadow: 0 8px 18px rgba(90, 65, 190, .25); font-size: 1.25rem; font-weight: 1000;
}
.arena-impact {
  position: absolute; z-index: 5; left: 50%; top: 52%;
  transform: translate(-50%, -50%); font-size: clamp(2.4rem, 7vw, 5rem);
  filter: drop-shadow(0 5px 8px rgba(40,35,79,.18)); animation: impact-pop .8s steps(2, end) infinite;
}
.arena-result {
  position: absolute; z-index: 4; left: 50%; bottom: 12px;
  transform: translateX(-50%); width: fit-content;
  max-width: 90%; padding: 9px 15px; color: #6248c5; background: #fff;
  border-radius: 999px; box-shadow: 0 8px 20px rgba(40,35,79,.15);
  text-align: center; font-weight: 900; opacity: 0;
}
.battle-log {
  display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px;
  margin-top: 16px; max-height: 190px; overflow-y: auto; padding: 2px;
}
.battle-log-item {
  display: flex; align-items: center; justify-content: space-between; gap: 8px;
  padding: 9px 10px; background: #fff; border: 1px solid #ebe8f5;
  border-radius: 16px; box-shadow: 0 6px 16px rgba(54,43,107,.08);
  font-size: .78rem; font-weight: 800; opacity: 0;
}
.battle-log-item strong {
  flex-shrink: 0; color: #6248c5; background: #f0edff;
  border-radius: 999px; padding: 4px 7px;
}
@keyframes battle-reveal { to { opacity: 1; } }
@keyframes lunge-right {
  0%, 15%, 55%, 100% { transform: translateX(0) rotate(0) scale(1); }
  27% { transform: translateX(24px) rotate(5deg) scale(1.05); }
  38% { transform: translateX(-6px) rotate(-3deg) scale(.98); }
}
@keyframes lunge-left {
  0%, 15%, 55%, 100% { transform: translateX(0) rotate(0) scale(1); }
  27% { transform: translateX(-24px) rotate(-5deg) scale(1.05); }
  38% { transform: translateX(6px) rotate(3deg) scale(.98); }
}
@keyframes impact-pop {
  0%, 100% { opacity: .55; scale: .8; }
  50% { opacity: 1; scale: 1.18; }
}
@media (max-width: 600px) {
  .battle-arena { min-height: 370px; }
  .battle-fighters { left: 2%; right: 2%; gap: .25rem; }
  .fighter-card { width: 39%; }
  .fighter-name { font-size: .68rem; }
  .fighter-word { font-size: .78rem; }
  .battle-log { grid-template-columns: 1fr; }
  .arena-sun-label, .beach-palm { font-size: 3.5rem; }
}
@media (prefers-reduced-motion: reduce) {
  .fighter-image, .arena-impact { animation: none !important; }
  .fighter-hp-fill, .arena-result, .battle-log-item, .hp-final {
    animation: none !important; opacity: 1 !important;
  }
}
</style>
"""


def _keyframes(name: str, checkpoints: list[tuple[float, int]]) -> str:
    frames = " ".join(
        f"{position:.2f}% {{ width: {value}%; }}"
        for position, value in checkpoints
    )
    return f"@keyframes {name} {{ {frames} }}"


def _attack_events(fight_log: list[dict]) -> list[dict]:
    if not fight_log or "attacker" in fight_log[0]:
        return fight_log

    attacks = []
    challenger_hp = opponent_hp = 100
    for old_round in fight_log:
        challenger_damage = int(old_round["challenger_damage"])
        opponent_hp = max(0, opponent_hp - challenger_damage)
        attacks.append(
            {
                "attacker": "challenger",
                "damage": challenger_damage,
                "challenger_hp": challenger_hp,
                "opponent_hp": opponent_hp,
            }
        )
        opponent_damage = int(old_round["opponent_damage"])
        challenger_hp = max(0, challenger_hp - opponent_damage)
        attacks.append(
            {
                "attacker": "opponent",
                "damage": opponent_damage,
                "challenger_hp": challenger_hp,
                "opponent_hp": opponent_hp,
            }
        )
    return attacks


def render_battle_arena(battle: dict, language: str) -> str:
    photos = {photo["id"]: photo for photo in battle["photo_rows"]}
    challenger_photo = photos[battle["challenger_photo_id"]]
    opponent_photo = photos[battle["opponent_photo_id"]]
    scene = battle["scene"]
    is_beach = scene == "beach"
    scene_name = "해변" if language == "ko" and is_beach else "숲" if language == "ko" else "BEACH" if is_beach else "FOREST"
    challenger_name = html.escape(battle["challenger"])
    opponent_name = html.escape(battle["opponent"])
    challenger_word = html.escape(
        challenger_photo["korean_word"] if language == "ko" else challenger_photo["class_name"]
    )
    opponent_word = html.escape(
        opponent_photo["korean_word"] if language == "ko" else opponent_photo["class_name"]
    )
    challenger_image = base64.b64encode(challenger_photo["image"]).decode("ascii")
    opponent_image = base64.b64encode(opponent_photo["image"]).decode("ascii")
    challenger_hp = max(0, min(100, int(battle["challenger_hp"])))
    opponent_hp = max(0, min(100, int(battle["opponent_hp"])))
    attacks = _attack_events(battle["fight_log"])
    attack_count = max(1, len(attacks))
    seconds_per_attack = 0.72
    battle_duration = attack_count * seconds_per_attack
    challenger_progress = [(0.0, 100)]
    opponent_progress = [(0.0, 100)]
    for attack_index, attack in enumerate(attacks):
        hit_position = (attack_index + 1) * 100 / attack_count
        challenger_progress.append(
            (hit_position, int(attack["challenger_hp"]))
        )
        opponent_progress.append(
            (hit_position, int(attack["opponent_hp"]))
        )
    challenger_progress.append((100.0, challenger_hp))
    opponent_progress.append((100.0, opponent_hp))
    battle_token = hashlib.sha1(
        str(battle.get("id", "battle")).encode("utf-8")
    ).hexdigest()[:10]
    challenger_animation = f"hp-challenger-{battle_token}"
    opponent_animation = f"hp-opponent-{battle_token}"
    hp_animation_style = f"""<style>
{_keyframes(challenger_animation, challenger_progress)}
{_keyframes(opponent_animation, opponent_progress)}
</style>"""
    trail_delay = 0.5
    result_delay = battle_duration + trail_delay
    winner_id = battle["winner_id"]
    if winner_id is None:
        result = "무승부!" if language == "ko" else "DRAW!"
    else:
        winner = challenger_name if winner_id == battle["challenger_id"] else opponent_name
        result = f"🏆 @{winner} 승리!" if language == "ko" else f"🏆 @{winner} WINS!"

    log_items = []
    for attack_index, attack in enumerate(attacks):
        attacker_is_challenger = attack["attacker"] == "challenger"
        attacker_name = challenger_name if attacker_is_challenger else opponent_name
        target_name = opponent_name if attacker_is_challenger else challenger_name
        damage = int(attack["damage"])
        reveal_delay = (attack_index + 1) * seconds_per_attack
        log_items.append(
            f'<div class="battle-log-item" style="animation: battle-reveal .25s ease {reveal_delay:.2f}s forwards;">'
            f'⚡ @{attacker_name} → @{target_name}<strong>-{damage} HP</strong></div>'
        )

    return f"""{ARENA_STYLE}
{hp_animation_style}
<div class="arena-shell">
  <div class="battle-arena {scene}">
    <div class="arena-topline">
      <span class="arena-tag">PHOTO FIGHT // {scene_name}</span>
      <span class="arena-tag">AUTO BATTLE</span>
    </div>
    <div class="arena-sun-label">{'☀️' if is_beach else ''}</div>
    <div class="beach-palm">{'🌴' if is_beach else ''}</div>
    <div class="forest-trees">{'<span>🌲</span><span>🌳</span><span>🌲</span>' if not is_beach else ''}</div>
    <div class="battle-fighters">
      <div class="fighter-card fighter-left">
        <span class="fighter-name">@{challenger_name}</span>
        <img class="fighter-image" src="data:image/jpeg;base64,{challenger_image}" alt="{challenger_word}">
        <span class="fighter-word">{challenger_word}</span>
        <div class="fighter-hp">
          <div class="fighter-hp-trail" style="width:{challenger_hp}%; animation: {challenger_animation} {battle_duration}s linear {trail_delay}s both;"></div>
          <div class="fighter-hp-fill" style="width:{challenger_hp}%; animation: {challenger_animation} {battle_duration}s linear forwards;"></div>
        </div>
        <div class="fighter-hp-text"><span class="hp-final" style="animation: battle-reveal .2s ease {result_delay}s forwards;">HP {challenger_hp}/100</span></div>
      </div>
      <div class="arena-vs">VS</div>
      <div class="fighter-card fighter-right">
        <span class="fighter-name">@{opponent_name}</span>
        <img class="fighter-image" src="data:image/jpeg;base64,{opponent_image}" alt="{opponent_word}">
        <span class="fighter-word">{opponent_word}</span>
        <div class="fighter-hp">
          <div class="fighter-hp-trail" style="width:{opponent_hp}%; animation: {opponent_animation} {battle_duration}s linear {trail_delay}s both;"></div>
          <div class="fighter-hp-fill" style="width:{opponent_hp}%; animation: {opponent_animation} {battle_duration}s linear forwards;"></div>
        </div>
        <div class="fighter-hp-text"><span class="hp-final" style="animation: battle-reveal .2s ease {result_delay}s forwards;">HP {opponent_hp}/100</span></div>
      </div>
    </div>
    <div class="arena-impact">💥</div>
    <div class="arena-result" style="animation: battle-reveal .25s ease {result_delay}s forwards;">{result}</div>
  </div>
  <div class="battle-log">{''.join(log_items)}</div>
</div>
""".strip()
