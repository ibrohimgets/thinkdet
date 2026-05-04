# Think-then-Detect Commonsense Results (Merged 8-GPU)

- n_samples: 780

## Overall

| condition | hit@0.5_top1 | hit@0.5_topk | mean_iou_top1 |
|---|---:|---:|---:|
| baseline_direct | 0.4000 | 0.6551 | 0.3807 |
| baseline_translated | 0.6487 | 0.7769 | 0.5878 |
| thinkdet_direct | 0.4064 | 0.6538 | 0.3870 |
| thinkdet_translated | 0.6487 | 0.7718 | 0.5888 |

## Per-Object (ThinkDet Translated, sorted by hit@0.5_top1)

| object | ttd_hit50 | direct_hit50 | baseline_hit50 | n |
|---|---:|---:|---:|---:|
| airplane | 1.0000 | 0.9000 | 0.9000 | 10 |
| banana | 1.0000 | 0.5000 | 0.5000 | 10 |
| bear | 1.0000 | 0.9000 | 0.9000 | 10 |
| bench | 1.0000 | 0.7000 | 0.7000 | 10 |
| cake | 1.0000 | 0.9000 | 0.9000 | 10 |
| cat | 1.0000 | 0.8000 | 0.8000 | 10 |
| donut | 1.0000 | 0.5000 | 0.5000 | 10 |
| elephant | 1.0000 | 1.0000 | 1.0000 | 10 |
| fire hydrant | 1.0000 | 1.0000 | 1.0000 | 10 |
| frisbee | 1.0000 | 0.5000 | 0.5000 | 10 |
| giraffe | 1.0000 | 0.5000 | 0.5000 | 10 |
| horse | 1.0000 | 0.8000 | 0.8000 | 10 |
| kite | 1.0000 | 0.4000 | 0.4000 | 10 |
| pizza | 1.0000 | 0.8000 | 0.8000 | 10 |
| sandwich | 1.0000 | 1.0000 | 1.0000 | 10 |
| sheep | 1.0000 | 0.1000 | 0.1000 | 10 |
| skateboard | 1.0000 | 0.2000 | 0.1000 | 10 |
| snowboard | 1.0000 | 0.0000 | 0.0000 | 10 |
| tennis racket | 1.0000 | 0.4000 | 0.4000 | 10 |
| umbrella | 1.0000 | 0.2000 | 0.2000 | 10 |
| broccoli | 0.9000 | 0.5000 | 0.5000 | 10 |
| bus | 0.9000 | 0.5000 | 0.5000 | 10 |
| dining table | 0.9000 | 0.8000 | 0.8000 | 10 |
| fork | 0.9000 | 0.3000 | 0.2000 | 10 |
| remote | 0.9000 | 0.8000 | 0.8000 | 10 |
| teddy bear | 0.9000 | 0.5000 | 0.5000 | 10 |
| baseball bat | 0.8000 | 0.8000 | 0.8000 | 10 |
| boat | 0.8000 | 0.3000 | 0.3000 | 10 |
| chair | 0.8000 | 0.2000 | 0.1000 | 10 |
| clock | 0.8000 | 0.7000 | 0.7000 | 10 |
| couch | 0.8000 | 0.8000 | 0.9000 | 10 |
| motorcycle | 0.8000 | 0.0000 | 0.0000 | 10 |
| skis | 0.8000 | 0.0000 | 0.0000 | 10 |
| suitcase | 0.8000 | 0.2000 | 0.2000 | 10 |
| toilet | 0.8000 | 0.3000 | 0.3000 | 10 |
| toothbrush | 0.8000 | 0.7000 | 0.7000 | 10 |
| book | 0.7000 | 0.5000 | 0.5000 | 10 |
| dog | 0.7000 | 0.5000 | 0.4000 | 10 |
| keyboard | 0.7000 | 0.0000 | 0.0000 | 10 |
| spoon | 0.7000 | 0.6000 | 0.6000 | 10 |
| traffic light | 0.7000 | 0.4000 | 0.4000 | 10 |
| zebra | 0.7000 | 0.5000 | 0.6000 | 10 |
| baseball glove | 0.6000 | 0.1000 | 0.1000 | 10 |
| bed | 0.6000 | 0.6000 | 0.6000 | 10 |
| bird | 0.6000 | 0.5000 | 0.4000 | 10 |
| bottle | 0.6000 | 0.3000 | 0.2000 | 10 |
| car | 0.6000 | 0.1000 | 0.1000 | 10 |
| cell phone | 0.6000 | 0.4000 | 0.4000 | 10 |
| oven | 0.6000 | 0.2000 | 0.2000 | 10 |
| potted plant | 0.6000 | 0.1000 | 0.1000 | 10 |
| surfboard | 0.6000 | 0.4000 | 0.4000 | 10 |
| tie | 0.6000 | 0.0000 | 0.0000 | 10 |
| wine glass | 0.6000 | 0.6000 | 0.6000 | 10 |
| cow | 0.5000 | 0.6000 | 0.6000 | 10 |
| hot dog | 0.5000 | 0.5000 | 0.5000 | 10 |
| laptop | 0.5000 | 0.1000 | 0.2000 | 10 |
| mouse | 0.5000 | 0.0000 | 0.0000 | 10 |
| sports ball | 0.5000 | 0.4000 | 0.3000 | 10 |
| stop sign | 0.5000 | 0.3000 | 0.3000 | 10 |
| train | 0.5000 | 0.4000 | 0.4000 | 10 |
| backpack | 0.4000 | 0.3000 | 0.3000 | 10 |
| refrigerator | 0.4000 | 0.1000 | 0.1000 | 10 |
| sink | 0.4000 | 0.2000 | 0.2000 | 10 |
| bowl | 0.3000 | 0.5000 | 0.4000 | 10 |
| microwave | 0.3000 | 0.1000 | 0.1000 | 10 |
| truck | 0.3000 | 0.3000 | 0.3000 | 10 |
| tv | 0.3000 | 0.3000 | 0.3000 | 10 |
| apple | 0.2000 | 0.0000 | 0.0000 | 10 |
| bicycle | 0.2000 | 0.0000 | 0.0000 | 10 |
| carrot | 0.1000 | 0.5000 | 0.5000 | 10 |
| cup | 0.0000 | 0.0000 | 0.0000 | 10 |
| handbag | 0.0000 | 0.3000 | 0.3000 | 10 |
| knife | 0.0000 | 0.3000 | 0.3000 | 10 |
| orange | 0.0000 | 0.0000 | 0.0000 | 10 |
| parking meter | 0.0000 | 0.0000 | 0.0000 | 10 |
| person | 0.0000 | 1.0000 | 1.0000 | 10 |
| scissors | 0.0000 | 0.1000 | 0.1000 | 10 |
| vase | 0.0000 | 0.1000 | 0.1000 | 10 |

## Sample Translations

| object | original | translated |
|---|---|---|
| person | someone you talk with . | cell phone |
| person | someone you ask questions . | cell phone |
| chair | seat for one person . | chair |
| giraffe | spotted animal seen on trips . | giraffe |
| zebra | wild animal behind a fence . | zebra |
| elephant | huge animal people see at zoos . | elephant |
| elephant | wild animal behind a fence . | elephant |
| bowl | round dish for food . | plate |
| couch | something used while watching shows . | remote control |
| pizza | baked food with toppings . | pizza |
| horse | animal people ride . | dog |
| horse | animal used for riding lessons . | horse |
| sandwich | quick meal between bread . | hot dog bun |
| car | something parked outside . | car |
| dining table | flat surface for dinner . | tray |
| motorcycle | vehicle people ride . | bicycle |
| motorcycle | something worn with a helmet . | helmet |
| cow | animal that gives milk . | giraffe |
| sheep | farm animal kept in groups . | sheep |
| bus | ride that stops often . | bus |
| teddy bear | soft toy for hugging . | teddy bear |
| teddy bear | comfort toy for children . | teddy bear |
| donut | treat people eat for breakfast . | donut |
| truck | something used to haul items . | car |
| suitcase | luggage for vacations . | suitcase |
| cup | something you drink from . | wine glass |
| cup | something kept in the kitchen . | bowl |
| cake | baked treat for birthdays . | cake |
| banana | soft fruit kids eat . | banana |
| broccoli | vegetable people steam . | vegetable |
