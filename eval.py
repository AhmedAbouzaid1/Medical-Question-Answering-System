def tagsanswer (question):
    question = question.lower()
    question = question.translate(str.maketrans('', '', string.punctuation))

    total_words = question.split()
    total_word_length = len(total_words)
    # print(total_word_length)

    total_sentences = tokenize.sent_tokenize(question)
    total_sent_len = len(total_sentences)

    tf_score = {}
    for each_word in total_words:
        each_word = each_word.replace('.', '')
        if each_word not in stop_words:
            if each_word in tf_score:
                tf_score[each_word] += 1
            else:
                tf_score[each_word] = 1
    tf_score.update((x, y / int(total_word_length)) for x, y in tf_score.items())

    # print(tf_score)

    def check_sent(word, sentences):
        final = [all([w in x for w in word]) for x in sentences]
        sent_len = [sentences[i] for i in range(0, len(final)) if final[i]]
        return int(len(sent_len))

    idf_score = {}
    for each_word in total_words:
        each_word = each_word.replace('.', '')
        if each_word not in stop_words:
            if each_word in idf_score:
                idf_score[each_word] = check_sent(each_word, total_sentences)
            else:
                idf_score[each_word] = 1

    # Performing a log and divide
    idf_score.update((x, math.log(int(total_sent_len) / y)) for x, y in idf_score.items())

    # print(idf_score)

    tf_idf_score = {key: tf_score[key] * idf_score.get(key, 0) for key in tf_score.keys()}

    # print(tf_idf_score)

    def get_top_n(dict_elem, n):
        result = dict(sorted(dict_elem.items(), key=itemgetter(1), reverse=True)[:n])
        return result

    res = get_top_n(tf_idf_score, 100)
    tags = list(res.keys())[:]

    query = "SELECT id FROM questiontags.tags WHERE tag = "
    for x in tags:
        query = query + "'" + x + "'" + " or tag = "

    query = query[:len(query) - 10]
    # print(query)
    mycursor.execute(query)

    myresult = mycursor.fetchall()

    Output = collections.defaultdict(int)

    for elem in myresult:
        Output[elem[0]] += 1

    # Printing output
    a = sorted(Output.items(), key=lambda x: x[1], reverse=True)[:3]
    res = []
    for x in a:
        q2 = df['question']
        q2 = q2[x[0] - 1]
        xx = check_similarity(question, q2)
        if (xx[0] != 'contradiction'):
            res.append(xx[1])

    index = res.index(max(res))
    index = a[index]
    # print(index[0])
    Ans = df["answer"]
    answer = Ans[index[0] - 1]

    return answer, index[0] - 1,

def accuracy():
    counter_max = 0
    for num, question in enumerate(df['question']):
        answer , index = tagsanswer(question)
        if index[0] - 1 == num:
            counter_max += 1


