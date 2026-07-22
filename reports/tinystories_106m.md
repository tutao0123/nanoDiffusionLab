# TinyStories 106M AR vs. Masked Diffusion

> AR perplexity and masked-diffusion denoising loss are objective-specific; do not compare them as the same metric.

| Metric | autoregressive | masked_diffusion |
|---|---|---|
| Parameters | 106,276,032 | 106,610,112 |
| Input tokens | 2,000,158,720 | 2,000,158,720 |
| Supervised targets | 1,998,205,440 | 1,000,126,478 |
| Elapsed hours | 3.19 | 2.55 |
| Mean training tokens/s | 242,869 | 287,053 |
| Peak memory MiB/GPU | 18,770 | 14,080 |
| Best validation loss | 1.2368 | 1.9635 |
| Validation loss | 1.2368 | 1.9635 |
| AR perplexity | 3.4446 | — |
| MDLM masked accuracy | — | 0.5908 |
| Git commit | a553bb1dcefaa40d1bd1630b6c840c18e9857710 | 4e7f98d42a7a89a519b06f019b0bed46c37cc578 |

## Fixed-seed samples

### autoregressive

```text
Once upon a time there was an old lady. She was very kind and loved helping all the animals and plants in her garden. Every day she would work in her garden and make sure the plants had enough love.

One day, the old lady was working in the garden when she spotted an unusual thing. It was a strange piece! She bent down and picked it up. She looked closely and saw it was actually some special kind. On the piece there were some bright colored butterflies.

The old lady smiled. She started to carefully put the strange piece in her pocket. She wanted to help the butterflies find the special piece, and make sure they had enough sunlight.

The old lady continued to work in the garden, looking for more special pieces, but she found none at all. She was starting to feel a little bit sad. Then she heard a voice.

"Hey, old lady!", it said. "It looks like the unusual piece I found in my garden!"

The old lady's heart was warm, and she knew that if she could find the special paper for them, the butterflies might need it. She smiled and handed them the paper.

The little ones thanked the old lady. She looked around, but there

--- sample ---

Once upon a time there were two best friends who liked to eat blueberries. One day they found a big bush with lots of juicy, blueberries, and they both wanted to eat them at the same time. They saw a group of small animals nearby and wanted to share the blueberries.
But one of the little friends said, ""No, these are my blueberries!"
The other little friend ignored this, so both one ran over and started eating the blueberries. The blueberries started to disappear and the little friend started to cry.
Finally, the blueberries were all gone and there were no more blueberries left.
The two friends were both very unhappy and they had nothing to eat.
The end.<|endoftext|>John and Jack were the best of friends. They were always together outside, playing and laughing. On one sunny day, John suggested that they try and make a snowman. Jack was a little frightened. He had never made a snowman before and didn't know what they would have for a long time. But John encouraged him and said, "It will be fun!"

They got to work, stacking snow with big balls. Jack started to make the snowman, but Jack still scared. He held the two

--- sample ---

Once upon a time there was a small rabbit who was feeling very anxious. He wanted something to make him feel better, so he decided to take a nap in his favourite spot in the meadow. When he lay down, he began to doze off.

When he woke up in the sunny meadow, he saw a big, loud, friendly dog standing next to him. The rabbit was surprised, for he had never seen a dog before!

He asked the dog, "What are you doing here?"

The dog replied, "Oh, it's my shadow. We can play tag together - and no one can hurt us."

The rabbit was so excited! He quickly grabbed his tail and began to run. The dog chased after him, but sadly it was no catch. By the time the sun was setting, the rabbit felt relieved and he returned home, ready for bed!<|endoftext|>Charlie and her mum were walking in the park. Charlie liked to look at all the different flowers and trees. She also liked to admire the bright yellow ones, the big yellow ones and the colourful ones.

They came across a big tree with a really bright yellow fruit at the bottom. But when they got closer they realized it

--- sample ---

Once upon a time, there was a mommy and her little girl. They wanted a special treat, so the mommy asked the little girl to close her eyes. The little girl did as she was told and the mommy pulled her over. Suddenly a big, yummy cupcake appeared in the girl's hands. It was so delicious; they both wanted to eat it!

The mommy said they had to wait, but the little girl was so excited and she jumped around and laughed. The mommy picked up the cake and handed it to the little girl. She was so happy that she was able to get a treat.

The mommy and the little girl walked together back home. When they got there, they gave the mommy a big, yummy cupcake. They enjoyed it together and the little girl started to count the days until finally she reached the birthday cake.<|endoftext|>Once upon a time there was a little boy. One day he asked his mom if he could have some pizza. His mom said it was okay as long as he was careful. So the little boy went to the kitchen to make his pizza. He got out some cheese, tomato and cheese. He put them in a bowl and then put them in
```

### masked_diffusion

```text
Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she went to go park to play with her friends. She saw a big slide and ran over.

When she was there, she saw a nut on the ground. She wanted it, but but knew it was too high. She tried to climb it, but it was too high for her She fell on the ground and and fell on the ground. She cried.

Suddenly, her mom came came and helped her up. They picked her up and Lily back off the tree. She went back to the park and have fun with her friends. The end.<|endoftext|>Once upon time.,Once upon a time, there was a big slide.. The was was very tall and shiny, but it was very high. One day, a little girl came to the park and saw the big slide. She saw it was big and and wanted to climb it.

The little girl climbed the slide and started to the down. She was so happy that she could't the ladder anymore.

After that, the little little girl sat on the slide and started to slide down the slide. SheOnce they went time, the little girl

--- sample ---

Once upon a time, there was a little girl named Lily. She loved to play outside her backyard every day.. One day, Lily's to asked her to clean up her toys. Lily was very happy and listened to her mom.


Later that day, Lily's mom asked her if clean up her toys first. Lily didn't want to clean up her toys, so she knew she want to clean her toys.

Her mom said, "Lily, you need to clean your toys your toys. You need to clean me and you need to clean clean up your toys."

Lily her Lily cleaned up her toys and put away her mom's room look tidy. Lily was happy and she listened to her mom. From that on,, she always cleaned up her toys and told her mom. The end.<|endoftext|>Once upon a time, there was a little girl named Lily. She loved to play with her toys and sing songs. One day, she mommy gave her a new toy toy to play with. Lily was so happy and excited to play with it.

But one day, Lily's mommy said her it was time for bed. Lily was sad because she went to bed. She missed her mommy

--- sample ---

Once upon a time, there was a little girl named Lily. She loved to play with her toys and teddy bears. One day, she went to the park to play with her her friends. She saw a big slide and ran to to it.


"Hello,
" asked.

"It's your name?"

"My name is," Lily asked.

"My name is Lily's,"," Lily said.


"My name is Lily," Lily said.

"My name is Lily," the boy said said.

"" name?"?"

"Yes name," the boy said. "Can I say hi?"


"Sure," said Lily boy.

"Hi
The boy said and they said friends. They took the boy played to the slide and slid on the slide. They the slide and had a fun day at the park.<|endoftext|>Once upon a time, little Lily was to go to the park to play with her friends. One day, she felt sad because she had lost her favorite toy. She looked everywhere and low, but she couldn't find it.

Herily asked her mom, help, but it was nowhere. Her mom asked that

--- sample ---

Once upon a time, there was a little girl named Lily. She loved to play with her toys, but her favorite was a pink teddy bear. One day, she went to the park with her friend, Timmy. She saw a big slide and wanted to climb it it.

Lily's Timmy, "Will the slide is too high?"

Tim said,, "Okay, but you careful too high."

"."

Timily and Tim went went to together. on the feet and slid down the slide. They climbed up the slide and went up slide down together. They reached the top of the slide and had lots of time together. The end.<|endoftext|>Once upon a time a time, there was a girl named Lily.. She loved to run and run around in the grass. One day, she saw a big butterfly and ran towards it. She saw the butterfly and wanted to chase after it. But she flew, but it was flying high in the sky.

L, wanted to catch the big again, but she knew it was too fast. She started to run and try to catch it. But then, she lost her grip and couldn't catch the ground. She was sad and
```
