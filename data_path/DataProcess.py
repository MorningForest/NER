import codecs

fp = codecs.open('./renmin3.txt', 'r', encoding='utf-8')
train_fp = codecs.open('./train_data', 'a+', encoding='utf-8')
test_fp = codecs.open('./test_data', 'a+', encoding='utf-8')
valid_fp = codecs.open('./valid_data', 'a+', encoding='utf-8')
data = [line for line in fp.readlines() if len(line.strip()) > 0]
nums = len(data)
test_num = int(nums*0.2)
valid_num = test_num
train_num = nums - test_num -valid_num
for line in data[0:train_num]:
    token = line.split()
    for item in token:
        item = item.split('/')
        if item[1] == 'B_nt':
            item[1] = 'B-ORG'
        if item[1] == 'M_nt' or item[1] == 'E_nt':
            item[1] = 'I-ORG'
        if item[1] == 'B_nr':
            item[1] = 'B-PER'
        if item[1] == 'M_nr' or item[1] == 'E_nr':
            item[1] = 'I-PER'
        if item[1] == 'B_ns':
            item[1] = 'B-LOC'
        if item[1] == 'M_ns' or item[1] == 'E_ns':
            item[1] = 'I-LOC'
        train_fp.write('{}	{}\n'.format(item[0], item[1]))
    train_fp.write('\n')
for line in data[train_num:test_num+train_num]:
    token = line.split()
    for item in token:
        item = item.split('/')
        if item[1] == 'B_nt':
            item[1] = 'B-ORG'
        if item[1] == 'M_nt' or item[1] == 'E_nt':
            item[1] = 'I-ORG'
        if item[1] == 'B_nr':
            item[1] = 'B-PER'
        if item[1] == 'M_nr' or item[1] == 'E_nr':
            item[1] = 'I-PER'
        if item[1] == 'B_ns':
            item[1] = 'B-LOC'
        if item[1] == 'M_ns' or item[1] == 'E_ns':
            item[1] = 'I-LOC'
        test_fp.write('{}	{}\n'.format(item[0], item[1]))
    test_fp.write('\n')
for line in data[-test_num:]:
    token = line.split()
    for item in token:
        item = item.split('/')
        if item[1] == 'B_nt':
            item[1] = 'B-ORG'
        if item[1] == 'M_nt' or item[1] == 'E_nt':
            item[1] = 'I-ORG'
        if item[1] == 'B_nr':
            item[1] = 'B-PER'
        if item[1] == 'M_nr' or item[1] == 'E_nr':
            item[1] = 'I-PER'
        if item[1] == 'B_ns':
            item[1] = 'B-LOC'
        if item[1] == 'M_ns' or item[1] == 'E_ns':
            item[1] = 'I-LOC'
        valid_fp.write('{}	{}\n'.format(item[0], item[1]))
    valid_fp.write('\n')
fp.close()
train_fp.close()
test_fp.close()
valid_fp.close()
