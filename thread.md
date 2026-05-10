To view keyboard shortcuts, press question mark
View keyboard shortcuts


Post

See new posts
Conversation
left curve dev
@leftcurvedev_
Anyone with 8GB or 12GB VRAM setups needs to understand that "-ncmoe" is the key flag to boost performance on llama.cpp

Here are my results for Qwen3.6 35B A3B, with 64k q8_0 context on a 8GB RTX 3070Ti:

⚪️ no flag → 8.7 tok/s
RAM: 13.6GB & VRAM: 7.8GB

🔴 -ncmoe 35 → 27.5 tok/s
RAM: 12.1GB & VRAM: 4.3GB

🟢 -ncmoe 30 → 32.5 tok/s
RAM: 12GB & VRAM: 5.6GB

🔵 -ncmoe 25 → 40.9 tok/s
RAM: 12GB & VRAM: 6.9GB

Please note the ram and vram usage you see are total usage of a windows pc, with the model running. My friend's setup: 8GB VRAM and 16GB RAM. You can boost performance by switching to Linux, just something to keep in mind.

Basically, this flag keeps the MoE experts in the first X layers on your CPU + RAM, instead of eating all your VRAM straight away. This is a smart hybrid offload way that lets you run bigger models without OOM while keeping the rest on your GPU for speed.

As we can see on the data, there's a sweet spot. When we lower it from 35 to 25, speed bumps +50% because there are more layers on your GPU (look at the VRAM usage). The key here is to play around with the number and fit as much as possible on your VRAM, goal is to have 1GB/800MB headroom to avoid stress.

↓ server flags below

Quote
left curve dev
@leftcurvedev_
·
May 8
Today I’m doing some testing with the RTX 3070 Ti. Let’s see what we can fit in 8GB VRAM, I’ll split this into two parts:

1) Finding the sweet spot for the -ncmoe parameter for maximum speed on base llama.cpp

2) Trying Turboquant, DFlash and MTP integrations to either fit more x.com/leftcurvedev_/…
11:36 PM · May 8, 2026
·
155.3K
 Views
Relevant
View quotes

left curve dev
@leftcurvedev_
·
May 8
llama.cpp built from source
CUDA drivers 13.0
UD-IQ3_XXS GGUF from Unsloth

server command with flags:

/llama.cpp/build/bin/llama-server  \
-m Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf \
-ngl 99 \
-np 1  \ 
--flash-attn on  \
--cache-type-k q8_0 \
--cache-type-v q8_0 \  
--ctx-size 65536
Show more
From huggingface.co
left curve dev
@leftcurvedev_
·
May 8
Btw, left host at 0.0.0.0 but don’t do that boys, use it locally or use your tailscale ip directly 👍
left curve dev
@leftcurvedev_
·
May 9
More testing
Quote
left curve dev
@leftcurvedev_
·
May 9
I know some bros want to see what happens when we push the value even further, so here we go:

⚪️ -ncmoe 25 → 41.8 tok/s 
RAM: 12GB & VRAM: 6.9GB

🔴 -ncmoe 23 → 43.8 tok/s
RAM: 12.2GB & VRAM: 7.4GB

🟢 -ncmoe 21 → 38.6 tok/s
RAM: 12.4GB & VRAM: 7.8GB

🔵 -ncmoe 19 → 19.8  x.com/leftcurvedev_/…

Need More VRAM
@needmorevram
·
May 9
This is a massive win for 12GB card users. I'm curious, have you noticed any significant drop in perplexity or reasoning quality when you push the -ncmoe value higher to prioritize the speed boost?

With a 3060 12GB, this opens up a lot of possibilities.
left curve dev
@leftcurvedev_
·
May 9
this doesn’t affect the model quality in any way, just speed 👀
Jean Louis 🇺🇬 ☕ 让·路易
@jeanlouisug
·
May 9
I have just tried, so on RTX 3090 24 GB VRAM opposite is true, if I use -ncmoe flag, the more N argument I give the slower it goes (30-60 t/s), without -ncmoe flag I get 145 tokens/sec.
left curve dev
@leftcurvedev_
·
May 9
You are already fitting the whole model on your VRAM. This flag is for people who want to run bigger models on smaller cards only!
Relevant people
left curve dev
@leftcurvedev_
low iq, high vram — sharing local ai and coding stuff
Trending now
What’s happening
Entertainment · Trending
#Suriya
Entertainment · Trending
#ThalapathyVijay
Sports · Trending
#RCBvsMI
Trending in India
SPLITS REAL WINNER HIMANSHU
Show more
Terms of Service
 |
Privacy Policy
 |
Cookie Policy
 |
Accessibility
 |
Ads info
 |

More
© 2026 X Corp.