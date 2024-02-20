orginal_data_lines = open("../data/sp_relations/train.replace_ne.withpool", 'r').readlines()
orginal_set = set()
for line in orginal_data_lines:
    splits = line.strip().split('\t')
    question = splits[2].strip()
    orginal_set.add(question)

error_data_lines = open("./error_log/sp_error_result.txt", 'r').readlines()
error_set = set()
zero_shot = 0
other = 0
output_file = open("./error_log/error_zero_shot.txt", 'w')
output_file1 = open("./error_log/error_other.txt", 'w')
for line in error_data_lines:
    splits = line.strip().split('\t')
    question = splits[0].strip()
    goldr = splits[1].strip()
    predr = splits[2].strip()
    error_set.add(question)

    # print(right, "-", right / count)
    # print(wrong, '-', wrong / count)
print("length error set:", len(error_set))
count = len(error_set)
for q in error_set:
    if q not in orginal_set:
        zero_shot += 1
        output_file.write(q + "\n")
    else:
        other += 1
        output_file1.write(q + "\n")
print(zero_shot, ':', count, ':', zero_shot / count)
print(other, ':', count, ':', other / count)
