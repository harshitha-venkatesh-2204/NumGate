# Auditor's independent verdicts (made from note text + table before comparing to truth.csv).
# Each entry: row_id -> (verdict, auditor value at cited precision, arithmetic)
from pathlib import Path
import pandas as pd
V = {
 484:("W",1014.34,"PARTY BUNTING Revenue_LY=1014.34 != 1041.34"),
 490:("C",3952.80,"WHHTLH Revenue_LY=3952.80"),
 494:("W",13.92,"PACK OF 12 SKULL TISSUES Revenue_LW=13.92 != 344.84"),
 503:("C",3059.0,"3 HOOK YoY=(537.03/17.00-1)*100=3059.0"),
 513:("W",127.33,"JUMBO BAG TOYS Revenue_LW=127.33; 166.37 is Revenue_LY"),
 522:("C",20.0,"3 BABY GIFT SET TW 1916.91, LW 2397.15, (1916.91/2397.15-1)*100=-20.03 -> fell 20.0"),
 538:("W",636.92,"rest-88 sum 56048.59/88=636.916 -> 636.92 != 615.92"),
 562:("C",205,"PARTY BUNTING Units_LW=205"),
 566:("W",151.5,"EDWARDIAN PARASOL NATURAL YoY=(1018.49/404.92-1)*100=151.5; 115.9 is WoW"),
 567:("C",1018.49,"Revenue_TW=1018.49"),
 571:("W",164.3,"WOOD 2 DRAWER WoW=(674.01/255.05-1)*100=164.266 -> 164.3 != 164.2"),
 575:("W",1186.1,"SET 10 LIGHTS YoY=(522.69/40.64-1)*100=1186.15 -> 1186.1 != 1168.1"),
 591:("W",3873.7,"PINK AND LILAC WoW=(1825.90/45.95-1)*100=3873.67 -> 3873.7 != 3783.7"),
 593:("W",1,"GIANT BLACK SUNGLASSES Orders_TW=1; 2 is Orders_LW"),
 600:("W",926.8,"PINK 3 PIECE WoW=(1039.65/101.25-1)*100=926.8 != 962.8"),
 602:("W",1506.00,"PLEASE ONE PERSON Revenue_LW=1506.00 != 1192.80"),
 649:("W",-13.2,"WHHTLH WoW=(1961.04/2259.40-1)*100=-13.2; 'rose' is wrong direction"),
 652:("C",1115.45,"RED RETROSPOT JUMBO BAG Revenue_TW=1115.45"),
 656:("C",18.27,"WOODEN ROUNDERS 584.76/32=18.274 -> 18.27"),
 670:("C",792,"PACK OF 72 Units_TW=792"),
 675:("C",572.06,"PIZZA SLICE DISH Revenue_TW=572.06"),
 682:("C",115,"CAKE STAND LOVEBIRD 345.80/3=115.27 -> ~115"),
 382:("C",-46.3,"WHHTLH WoW=(2009.77/3740.98-1)*100=-46.28 -> -46.3"),
 386:("W",874,"COOK WITH WINE Units_TW=874; 24 is Units_LW"),
 411:("C",1963.57,"7 DOOR MAT rows Revenue_LW sum=1963.57"),
 424:("C",46.7,"2 HANGING HEART rows TW 2957.77 LW 5548.89 -> -46.696 -> down 46.7"),
 427:("C",20,"DOOR MAT UNION FLAG Orders_TW=20"),
 443:("W",4.9,"KASHMIR units 498/101=4.931 -> 4.9x != 5.9x"),
 462:("C",1017.04,"DOOR MAT WELCOME PUPPIES Revenue_TW=1017.04"),
 464:("C",811.10,"HOME SWEET HOME METAL SIGN Revenue_TW=811.10"),
 472:("C",513,"SOMBRERO Units_TW=513"),
 301:("C",13.50,"DOORMAT HEARTS Revenue_LY=13.50"),
 334:("C",47,"SET OF 3 CAKE TINS Orders_TW=47"),
 342:("C",876.84,"CHARLOTTE BAG SUKI Revenue_TW=876.84"),
 350:("C",113.01,"SPACEBOY 904.05/8=113.006 -> 113.01"),
 362:("C",942.30,"DOORMAT FAIRY CAKE Revenue_TW=942.30"),
 363:("C",628.6,"DOORMAT FAIRY CAKE WoW=628.6 > 600 (qualifier over)"),
 1:("C",130586.63,"UK Revenue_LW=130586.63"),
 2:("C",10.1,"UK WoW=(143748.77/130586.63-1)*100=10.08 -> 10.1"),
 6:("C",3,"Channel Islands Orders_TW=3"),
 20:("W",160,"France Units_LW=160 != 106"),
 21:("C",1107.38,"Germany Revenue_TW=1107.38"),
 23:("W",0.7,"Germany share 1107.38/154660.65*100=0.716 -> 0.7 != 0.8 (0.8 is France)"),
 42:("C",-62.5,"Germany WoW=(1107.38/2953.98-1)*100=-62.51 -> down 62.5"),
 45:("W",659.43,"EIRE LW per order 2637.72/4=659.43; 492.33=Germany LW 2953.98/6"),
 49:("C",349.8,"France WoW=(1271.21/282.60-1)*100=349.8"),
 53:("C",96600,"United Kingdom Units_TW=96600"),
 68:("W",334.5,"Channel Islands WoW=(3401.24/782.80-1)*100=334.50 -> 334.5 != 343.5"),
 70:("C",5591.70,"Germany LW 2953.98 + EIRE LW 2637.72 = 5591.70"),
 88:("C",2,"Netherlands Orders_TW=2"),
 89:("W",-28.4,"EIRE WoW=(1889.68/2637.72-1)*100=-28.36; 'rose' is wrong direction"),
 200:("C",2021.25,"DOOR MAT RED SPOT Revenue_TW=2021.25"),
 201:("C",576.44,"DOOR MAT RED SPOT Revenue_LW=576.44"),
 208:("C",47,"WHHTLH Orders_TW=47"),
 209:("C",45,"WHHTLH Orders_LW=45"),
 215:("C",6,"DOOR MAT SPOTTY HOME SWEET HOME Rank=6"),
 217:("C",3472.13,"table Revenue_LW sum=3472.13"),
 221:("C",6467.60,"SMALL FAIRY CAKE Revenue_TW=6467.60"),
 248:("C",66.1,"top-3 TW 10435.18/15781.74=66.12 -> 66.1"),
 256:("C",46,"DOOR MAT SPOTTY HOME SWEET HOME Units_LW=46"),
 267:("C",1946,"PAISLEY MUG Revenue_TW=1946.33 -> ~1946"),
 280:("C",21.8,"bottom-2 TW 3444.69/15781.74=21.83 -> 21.8"),
 103:("C",-39.9,"PAPER CHAIN 50'S WoW=(5701.16/9485.15-1)*100=-39.89 -> -39.9"),
 129:("C",5701.16,"PAPER CHAIN 50'S Revenue_TW=5701.16"),
 133:("W",13995.47,"2 PAPER CHAIN KIT LW 9485.15+4510.32=13995.47; 12102.54=9485.15+2617.39 (ROTATING SILVER ANGELS)"),
 140:("C",229.3,"CHILLI LIGHTS YoY=(4960.25/1506.40-1)*100=229.28 -> 229.3"),
 147:("C",37,"ROTATING SILVER ANGELS Orders_TW=37"),
 175:("C",105,"RABBIT NIGHT LIGHT Orders_TW=105"),
 187:("C",-32.7,"PAPER CHAIN VINTAGE WoW=(3034.33/4510.32-1)*100=-32.72 -> -32.7"),
 190:("C",1170,"ROTATING SILVER ANGELS Units_TW=1170"),
}
base = str(Path(__file__).resolve().parents[1]) + "/"
s = pd.read_csv(base + "audit/sample.csv")
assert set(V) == set(s.row_id), set(s.row_id) ^ set(V)
s["aud_label"] = s.row_id.map(lambda i: V[i][0]); s["aud_value"] = s.row_id.map(lambda i: V[i][1]); s["aud_arith"] = s.row_id.map(lambda i: V[i][2])
s.to_csv(base + "audit/sample_verdicts.csv", index=False)
print("sampled", len(s), "recorded W", (s.label=="W").sum(), "auditor W", (s.aud_label=="W").sum())
d = s[s.label != s.aud_label]
print("label disagreements", len(d)); print(d[["row_id","note_id","line","value_text","label","aud_label","true_value","aud_value","expr"]].to_string())
# compare true_value too
import numpy as np
s["tv_diff"] = [not np.isclose(abs(a), abs(b), atol=p/2+1e-9) if pd.notna(a) else True for a,b,p in zip(s.true_value, s.aud_value, s.precision)]
print(s[s.tv_diff][["row_id","note_id","line","value_text","label","true_value","aud_value","precision","error_type","expr"]].to_string())
print(s[s.label=="W"][["row_id","note_id","value_text","true_value","aud_value","error_type","error_src"]].to_string())
