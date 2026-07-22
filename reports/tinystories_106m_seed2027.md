# TinyStories 106M AR vs. Masked Diffusion

> AR perplexity and masked-diffusion denoising loss are objective-specific; do not compare them as the same metric.

| Metric | autoregressive | masked_diffusion |
|---|---|---|
| Parameters | 106,276,032 | 106,610,112 |
| Input tokens | 2,000,158,720 | 2,000,158,720 |
| Supervised targets | 1,998,205,440 | 1,000,253,986 |
| Elapsed hours | 2.93 | 2.59 |
| Mean training tokens/s | 244,840 | 289,385 |
| Peak memory MiB/GPU | 18,770 | 13,873 |
| Best validation loss | 1.2409 | 1.9705 |
| Validation loss | 1.2409 | 1.9705 |
| AR perplexity | 3.4586 | — |
| MDLM masked accuracy | — | 0.5902 |
| Git commit | 4e7f98d42a7a89a519b06f019b0bed46c37cc578 | 4e7f98d42a7a89a519b06f019b0bed46c37cc578 |

## Fixed-seed samples

### autoregressive

```text
Once upon a time, there was a little fox. He was walking in a forest and saw a beautiful flower. It was red and yellow and looked so amazing.

The little fox bent down and asked the flower, "What are you doing?" The flower said nothing, simply sitting on the green grass.

The little fox thought for a moment and said to the flower, "I just wanted to see what the flower's name is and I think it's the most impressive thing I have ever answered. I'm so glad you came by just that."

After that, the little fox and the flower were best friends. They took a walk in the forest, looking for more amazing things to see and explore. They laughed and played together in the forest all day.<|endoftext|>Once upon a time, there was a little boy who was very weak. He couldn't do many things with his body, like sit down and sleep. He wanted to gain strength so he decided to try something new.

So, the little boy decided to take a risk and set out walking into a park. He felt brave and excited, but also scared of the feeling of the wind on his face. He started to walk, but soon he realized he was too weak.

--- sample ---

Once upon a time, there was a big race. All the racers were very excited because it was their favourite event.

The race was happening and the kids lined up at the starting line. The race was very long and the kids were very nervous. Suddenly, the racers started running faster and faster on the track.

At the end, the racers won the race! All the racers cheered and celebrated. They were happy to be together and celebrate, and they all enjoyed the event together.<|endoftext|>Once there was a little boy who loved to explore new places. He loved to wander around and discover new things. One day he decided to go and see what he could find.

He walked through the grass and eventually found a giant tree with lots of yummy fruits. He thought it was just great to see the fruits so bright and taste so sweet. He was so excited he couldn't stop looking around and looking for more.

He kept wandering around and found a box full of apples. He was so hungry that he couldn't even eat the apples when he had left the tree for the first time. He decided to take some of the apples home with him and share them with his family.

He knew

--- sample ---

Once upon a time there was a little bear who lived in a chamber. He loved to sleep a lot and play all day. One day, he heard a noise - it sounded like someone was calling his name. As the bear got out of his bed, he saw a mouse running by.

The mouse was scared and ran to hide. The bear followed him and finally arrived. He saw that the mouse was very scared and started to pet him. The mouse was so relieved and started to purr.

The bear smiled and started to talk to the mouse.
"Good morning!" said the bear. "You must be sleepy too, so come over. I'm going to eat something!"

So the mouse hopped off to the kitchen. The bear went to his chamber and started to feed the mouse. He gave him a few pieces of food and the mouse happily ate it all up.

The bear waved goodbye to the mouse and said, "It looks like you're going to be hungry again. Please come back and feed your next time!"<|endoftext|>Mama and Papa were outside with their three year old grandson, Tommy. Mama was always teaching him about the world and what he didn't understand.

Tommy had just started playing

--- sample ---

Once upon a time, there was a chubby bird named Blue. Blue loved to fly around and sing songs. One day, Blue saw his friend Blue walking near a fire. Blue wanted to play with the fire, but Blue didn't know how to share. Blue started to complain, "These fire is too big for me to play with. Let's play together instead." Blue understood and they played happily together.<|endoftext|>Once upon a time, there was a little girl named Lily. She loved to play in the garden and watch the flowers grow. One day, she heard a strange noise coming from the shed.

Lily went to investigate and found a little squirrel stuck up a rope. "I can make it jump!" she said.

But just as she was about to grab it, a big, mean dog came running towards her. "Uh-oh," said Lily, feeling helpless.

The squirrel quickly jumped into her arms and caught her. From that day on, Lily always kept an eye out for the little squirrel. And whenever she heard the loud noise from the shed, she remembered the squirrel and waved her hands at him to let him know he was there.<|endoftext|>Once upon a time, there was a little boy named Timmy
```

### masked_diffusion

```text
Once upon a time, there was a girl named Jack. He was three years old. He had a years old. Jack was thoughtful. One day, she wanted to show her present to her friend. She gave him a present and a big of paper. Jack was amazed!

She asked him how to fold it. Jack. She showed him how to fold the paper.

She showed Jack how to fold paper. She showed Jack how to fold paper. She smiled.

Jack was so excited. He wanted to show his present. He

He took his piece of paper and showed how to fold the paper. He folded it to and soon it worked.

Jack showed her present to her who was very impressed. They were him impressed see the present.. He was so proud of Jack and the present. He was so proud of himself for folding the present. He was so proud of her present.<|endoftext|>Once upon time, there was a boy. He was a happy. He He loved to go to in the sea.

One day, he decided to to the beach. He put on his shoes and jumped in the waves.

He said goodbye to the beach beach. He spl around and in the water. He

--- sample ---

Once upon a time, there was a little girl named Tim. Sue. She loved to to play with the ball. One day, while playing, he saw a big, red ball near the fence. He wanted to get the ball, but it was too high.

Sue was friend, "Can I get the ball, Tim?" asked him?" Tim said, "I want't reach the ball ball." but he was sad.. He showed Sue an ball. She was worried. She said, "I can help you if you reach my it."

They looked around for the ball. They found the ball ball and then went to the ball. Tim finally got the ball and ran it. Sue was so happy. She said "Thank you, Timmy! You are the best my friends!" They played with the ball all day.<|endoftext|>Once upon time,, there was a man. The man lived in a house. The house. He lived in a big house. The man was very happy.

The day, the man went outside. He saw the colors. It was very pretty. He wanted to see the colors the colors.

The next day, the man went outside. So. He went to the park.

--- sample ---

Once upon a time, there a brave little little girl. She was three three old and she loved to explore. She loved to explore and discover new things. One day, she decided to explore a big tree in the woods. She started to climb the tree. She was very brave. She wanted to climb the tree. but she she to was brave.

She got the top climb the tree. She was very proud and herself. After a while, she was at the top. She smiled. She was so brave. She was so brave.

She climbed up the tree. She looked at the top of the tree. Nothing. She was brave. She was very brave.

She had reached at the tree. She was so proud. Her bravery was She that she was brave. She climbed down the tree and went back home. He was happy.<|endoftext|>John was sad. He was sad. He was happy. He saw the toy. It was a toy. John. He went to the it.

He looked around. He saw something...
John went to the store. He saw the toy. He. He smiled. He just smiled. He wanted the toy..

John left. He was sad.

--- sample ---

Once upon a time, there was a little girl named Lily. She was very happy and loved to play with her friends. One day, Lily's friends friends came to play outside to play... Lily was to listen to them and wanted to help her friends. So.

Lily and her friends went outside and joined her friends. She played on the swings and had lots of fun. They ran and jumped and laughed a lot. Lily.

But playing, it was a storm came. The wind was hard and strong... Lily's other friends came and came to help her. They went to the park and saw a tree. She wanted to climb up the tree.

Lily told her friends the tree about climbing pretty tree. They climbed around and tree. They talked on the birds. The bird came and came them from the tree. Lily was happy too. She decided to share and play with her friends. They all laughed and played with the park. Lily was very happy.<|endoftext|>Once upon a time there was a bird.. One day, it saw a pretty bird in the sky. The bird was happy and saw the bird. The bird knew the bird wanted to eat the bird. The bird was happy and the bird
```
