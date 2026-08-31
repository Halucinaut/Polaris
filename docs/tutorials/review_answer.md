### S1：

1. target每一条都是\<think>开头，\</think>结尾，然后跟上\n\n\\\boxed{}来标记答案。
2. metadata.answer与boxed answer完全一致
3. reasoning中没有<<或者###残留，没有计算错误。
4. system prompt这五百条完全一致。

   总结，可以说明目前这500条的数据是干净的。

### S2

1. `logits[:, :-1]对应的input position应该是0,1,2,3`
2. `labels[:, 1:]的值是-100，-100，THINK，EOS`
3. 位置2和3，即THINK和EOS的预测进入loss
4. 如果误用，相当于用t时刻的logits预测t，即没有实现t+1的预测监督

<br />

### S3

1. input position为94的assistant的token id被设定为-100，是合理的，即prompt本身不受监督。
2. position 95 的 logits 预测 `<think>`
3. `<think>` 之后的 target token 持续受监督
4. 但是我没在shift check里看到eos

<br />

### S4

```python
steps_per_epoch = max(1, len(tokenized) // batch_size) # 即500//4 = 125
effective_steps = int(train_cfg.get("num_epochs", 1)) * steps_per_epoch # 所以 = 3*125 = 375
```

1. 得到375个optimizer steps
2. 每个step处理4条样本，循环四次，所以消耗16条样本。
3. 样本位置总共为375\*✖️4✖️4，即6000\*
4. 约等于完整遍历12次
5. 不准确，低估了实际训练量
   - num\_epochs=3 只描述了数据被顺序遍历的次数
   - 但没有考虑梯度累积（ grad\_accum=4 ）
   - 实际上数据被完整遍历了 3 × 4 = 12 次
   - 更准确的描述应该是： 有效遍历次数 = num\_epochs × grad\_accum = 12 次

### S5

1. A：抽取结果正确，抽取方法为box模式，格式遵循正确，答案正确
2. B：抽取结果正确，方法为最后的数字，格式遵循不正确，答案正确
3. C：抽取结果不正确，方法为box模式，格式遵循不正确，答案正确
4. D：抽取结果不正确，方法为最后数字，格式遵循不正确，答案正确
5. E：抽取结果正确，方法为box模式，格式遵循正确，答案不正确

<br />

### S6

因为run的内部评估数据量不足，容易有统计偏差，且各项参数配置不一定是标准的。模型应该是qwen3-0.6B，优化器是500样本的SFT的lora，数据集是GSM8K，样本数是500。可靠结论应该是SFT后，精度提升了8个点。

<br />

###

