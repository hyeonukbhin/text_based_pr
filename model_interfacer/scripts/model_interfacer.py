#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import gensim
from sklearn.externals import joblib
import rospy
from std_msgs.msg import String
# from pprintpp import pprint
import json
import rospkg

PACKAGE_PATH = rospkg.RosPack().get_path("model_interfacer")
MODEL_FILEPATH = PACKAGE_PATH + "/scripts/models/"

def read_df(filename):
    df = pd.read_csv(filename, sep=',', na_values=".", index_col=0)
    return df

def map_list_type(l, dtype=str):
    return list(map(dtype, l))

def get_name(label):
    if label == 0:
        name = "\x1b[1;31m{}\x1b[1;m".format(" ||     ")
    elif label == 1:
        name = "\x1b[1;32m{}\x1b[1;m".format(" ||||   ")
    elif label == 2:
        name = "\x1b[1;34m{}\x1b[1;m".format(" |||||| ")
    else:
        name = ""

    return name

def get_name_lmh(label):
    if label == 0:
        name = "low"
    elif label == 1:
        name = "middle"
    elif label == 2:
        name = "high"
    else:
        name = ""

    return name

def print_tui(lr_result):
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("================================================")
    print("")
    print("\x1b[1;33m{}\x1b[1;m".format("Doc2Vec Features + Gradient Boosting Classifier"))
    print("")
    print("")
    print("")
    print("{} {} {} {} {}".format(get_name(lr_result[0]),get_name(lr_result[1]),get_name(lr_result[2]),get_name(lr_result[3]),get_name(lr_result[4])))
    print("{} {} {} {} {}".format(get_name(lr_result[0]),get_name(lr_result[1]),get_name(lr_result[2]),get_name(lr_result[3]),get_name(lr_result[4])))
    print("[ EXT. ] [ NEU. ] [ AGR. ] [ CON. ] [ OPN. ]")
    print("[외향성] [신경성] [친화성] [성실성] [개방성]")
    print("")
    print("================================================")
    # print(lr_result)
    # print(gb_result)

def personality_model(tokens):
    # 모델 가져오기
    model_doc2vec = gensim.models.Doc2Vec.load(MODEL_FILEPATH+"personality_doc2vec.model")

    test_x = model_doc2vec.infer_vector(tokens)

    model_lr_classfier_ext = joblib.load(MODEL_FILEPATH+"lr_ext.sav")
    model_lr_classfier_neu = joblib.load(MODEL_FILEPATH+"lr_neu.sav")
    model_lr_classfier_agr = joblib.load(MODEL_FILEPATH+"lr_agr.sav")
    model_lr_classfier_con = joblib.load(MODEL_FILEPATH+"lr_con.sav")
    model_lr_classfier_opn = joblib.load(MODEL_FILEPATH+"lr_opn.sav")

    lr_result = [model_lr_classfier_ext.predict([test_x])[0],
                 model_lr_classfier_neu.predict([test_x])[0],
                 model_lr_classfier_agr.predict([test_x])[0],
                 model_lr_classfier_con.predict([test_x])[0],
                 model_lr_classfier_opn.predict([test_x])[0]
                 ]

    model_gb_classfier_ext = joblib.load(MODEL_FILEPATH+"gb_ext.sav")
    model_gb_classfier_neu = joblib.load(MODEL_FILEPATH+"gb_neu.sav")
    model_gb_classfier_agr = joblib.load(MODEL_FILEPATH+"gb_agr.sav")
    model_gb_classfier_con = joblib.load(MODEL_FILEPATH+"gb_con.sav")
    model_gb_classfier_opn = joblib.load(MODEL_FILEPATH+"gb_opn.sav")

    gb_result = [model_gb_classfier_ext.predict([test_x])[0],
                 model_gb_classfier_neu.predict([test_x])[0],
                 model_gb_classfier_agr.predict([test_x])[0],
                 model_gb_classfier_con.predict([test_x])[0],
                 model_gb_classfier_opn.predict([test_x])[0]
                 ]

    result = lr_result
    return result

def send_recognition(name, result):
    current_time = rospy.get_rostime()
    msgs_dict = {
        "header": {
            "timestamp": "%i.%i" % (current_time.secs, current_time.nsecs),
            "source": "perception",
            "target": ["planning", "action"],
            "content": ["human_personality"]
        },
        "human_personality": {
            "name": name,
            "extraversion": get_name_lmh(result[0]),
            "neuroticism": get_name_lmh(result[1]),
            "agreeableness": get_name_lmh(result[2]),
            "Conscientiousness": get_name_lmh(result[3]),
            "openness": get_name_lmh(result[4])
        }
    }
    json_string = json.dumps(msgs_dict, ensure_ascii=False, indent=4)
    # json_string = json.dumps(msgs_dict, ensure_ascii=True, indent=4)
    pub_recognition.publish(json_string)
    # pub_intent_topic.publish("안녕하세요")

    # print(msgs_dict)


def get_header(json_dict):
    source = json_dict["header"]["source"]
    target_list = json_dict['header']["target"]
    content_list = json_dict['header']["content"]

    return source, target_list, content_list

def callback_document(data):
    # json_dict = json.loads(data.data.decode('utf-8'))
    json_dict = json.loads(data.data)
    source, target_list, content_list = get_header(json_dict)
    # 사람 위치 추적 및 이름 파라미터 업데이
    if ("perception" in target_list) and (source == "perception") and ("document_result" in content_list):

        name = json_dict["document_result"]["name"]
        tokens = json_dict["document_result"]["tokens"]

        # 204, "['tryin', 'figure', 'w', 'plane', 'ticket', 'NYC', '24th', 'w', 'hang', 'FML', 'lets', 'drunk', 'somebody', 'livin', 'dream', 'Findin', 'steady', 'pay', 'check', 'year', 'Lovin', 'Hawaiian', 'pizza', 'I', 'wish', 'I', 'kiss', 'Because', 'I', 'know', 'taste', 'betrayal', 'WHITEOUT', 'credit', 'card', 'roulette', 'cost', '8110FML', 'Im', 'better', 'ya', 'ex', 'gon', 'na', 'better', 'ya', 'PSU', 'loss', 'Steelers', 'lossmy', 'life', 'shambles', 'IM', 'football', 'Southsidelife', 'good', 'insert', 'depressing', 'emo', 'lyrics', 'PSU', 'weekend', 'obvi', 'dont', 'think', 'I', 'dont', 'think', 'dont', 'think', 'I', 'dont', 'regrets', 'Maybe', 'break', 'better', 'Football', 'daythats', 'I', 'love', 'Sundays', 'Lovin', 'life', 'shortys', 'like', 'melody', 'head', 'Getting', 'needed', 'TLC', 'momand', 'trap', 'cleaner', 'nice', 'INDEPENDENT', 'know', 'means', '801', 'You', 'dropped', 'wrong', 'girls', 'house', '1801', 'Theres', 'thing', 'wrong', 'girl', 'happen', '801', 'alright', 'morning', 'CMU', 'footballis', 'better', 'ugh', 'rough', 'weekend', 'dying', 'dysentry', 'WPT', 'yes', 'terrible', 'night', 'Southside', 'tonightany', 'takers', 'boot', 'ass', 'American', 'way', 'I', 'got', 'feelin', 'tonights', 'gon', 'na', 'good', 'night', 'This', 'man', 'cave', 'theres', 'women', 'allowed', 'I', 'got', 'jerkoff', 'station', 'Gods', 'sake', 'Deceiving', 'Thats', 'world', 'calls', 'romance', 'Taking', 'frustrations', '300', 'plus', 'pounds', 'gravity', 'motivated', 'iron', 'I', 'know', 'got', 'reason', 'past', 'away', 'Tattoo', 'lower', 'backmight', 'bullseye', 'Just', 'trying', 'play', '1', 'round', 'golf', 'summer', 'thought', 'chick', 'hitting', 'night', 'turned', 'getting', 'attention', 'fun', '20', 'wearing', 'Abercrombie', 'FML', 'job', 'sparkling', 'wiggles', 'I', 'love', 'PSU', 'day', 'work', 'fin', 'right', 'doggie', 'Round', '2', 'Ive', 'better', 'living', 'hazelooking', 'meaning', 'life', 'finding', 'darkness', 'trying', 'justify', 'CMU', 'education', 'exchange', 'PSU', 'atmospherenot', 'working', 'cooler', 'online', 'trust', 'lies', 'core', 'love', ';', 'true', 'love', 'trust', 'working', 'week', 'time', 'play', 'gon', 'na', 'little', 'bit', 'sideways', 'wicked', 'salacity', 'leads', 'anger', 'frustration', 'youre', 'sure', 'love', 'youre', 'sure', 'let', 'taking', 'step', 'forward', '2', 'steps', 'backstory', 'life', 'lonely', 'night', 'hold', 'memories', 'better', 'days', 'exhausted', 'recovery', 'day', 'ugh', 'rough', 'shape', 'Dont', 'worry', 'people', 'past', 'theres', 'reason', 'didnt', 'future', 'said', 'probably', 'dont', 'remember', 'I', 'said', 'memory', 'like', 'burning', 'end', 'midnight', 'cigarette', 'Trying', 'learn', 'build', 'cowboy', 'hat', 'Natty', 'light', 'boxes', 'noddin', 'head', 'like', 'Yeah', 'movin', 'hips', 'like', 'Yeah', 'Just', 'got', 'pizza', 'delivered', 'house', 'Pittsburgh', 'chick', 'I', 'went', 'high', 'school', 'Coudersportfeel', 'better', 'life', 'right', 'I', 'dont', 'feel', 'like', 'watching', 'football', 'I', 'watch', 'good', 'moviePROPNAMEmanhood', 'question', 'Jammin', 'PROPNAME', 'new', 'Trapp', 'piano', 'All', 'classy', 'broads', 'hit', 'half', 'price', 'wine', 'night', 'right', 'anotha', 'day', 'anotha', 'dolla', 'hang', 'tight', 'cause', 'gon', 'na', 'wilder', '8', 'second', 'ride', 'ridiculous', 'FML', 'single', 'youre', 'willing', 'try', 'hard', 'PHI', 'callin', 'Ready', 'long', 'weekend', 'false', 'alarm', 'invincible', 'State', 'blows', 'opportunityshouldnt', 'expected', 'I', 'officially', 'dumbest', 'person', 'entire', 'worldmy', 'life', 'sucks', 'Chitown', 'weekend', 'False', 'alarmstill', 'invincible', 'Is', 'May', 'Turkey', 'Bowl', 'Beaver', 'Stadiumsick', 'guess', 'thats', 'youre', 'Tailgating', 'Wendys', 'parking', 'lot', 'new', 'thing', 'Composing', 'script', 'greek', 'singwill', 'definitely', 'epic', 'LMFAO', 'tryin', 'streak', 'alive', 'To', 'feel', 'pain', 'feel', 'Go', 'inside', 'eachothers', 'minds', 'wed', 'find', 'Look', 'shit', 'eachothers', 'eyes', 'Man', 'date', 'HokkaidoSMAAART', 'scars', 'heal', 'glory', 'fades', 'left', 'memories', 'pain', 'hurts', 'minute', 'life', 'short', 'live', 'cause', 'chicks', 'dig', 'needs', 'formal', 'date', 'Friday', 'nightstat', 'Cleveland', 'celebrating', 'birthdays', 'King', 'New', 'Years', 'burg', 'cruisin', 'Caribbeangonna', 'nice', 'little', 'weekend', 'Im', 'pretty', 'sure', 'theres', 'lot', 'life', 'ridiculously', 'good', 'looking', 'And', 'I', 'plan', 'finding', '85', 'sunny', 'Jamaica', '82', 'sunny', 'Grand', 'CaymanSMICK', 'Puzzling', 'face', 'Women', 'lie', 'men', 'lie', 'numbers', 'dont', 'lie', '4', 'rounds', 'credit', 'card', 'roulette', 'shots', 'paylove', 'Trapp', 'mandatedinner', 'movie', 'better', 'Lately', 'Ive', 'hard', 'reach', 'Ive', 'long', 'Everybody', 'private', 'world', 'Are', 'calling', 'Are', 'trying', 'Are', 'reaching', 'Im', 'reaching']", "['5.0', '1.75', '3.0', '3.5', '3.5']", Post_67
        # 0, "['CDC', 'latest', 'report', 'Autism', 'effects', '1', '110', 'children', 'born', 'day', '1', '70', 'boys', 'Put', 'status', '1', 'HOUR', 'know', 'AUTISM', 'Let', 'childrens', 'voices', 'heard', 'Heres', '2010', 'increasing', 'awareness', 'research', 'proactively', 'finding', 'answers', 'going', 'pool', 'later', 'writing', 'Psychology', 'essay', 'ew', 'Chemistry', 'art', 'special', 'going', 'anymore', 'better', 'I', 'guess', 'finishing', 'Psychology', 'essays', 'wants', 'cut', 'hair', 'finishing', 'homework', 'On', 'Algebra', 'II', 'going', 'art', 'class', 'writing', '50', 'page', 'mark', 'wants', 'keyboard', 'German', 'classs', 'sick', 'And', 'bored', 'Someone', 'talk', 'meee', 'starts', 'Psychology', 'tomorrow', 'haha', 'PROPNAME', 'awesome', 'xD', 'Chemistryy', 'soo', 'hardd', 'doin', 'German', 'homework', 'First', 'Mission', 'Decode', 'Paramore', 'goin', 'college', 'wants', 'kitten', 'I', 'shall', 'PROPNAME', 'sitting', 'outside', 'Pennsylvania', 'kittens', 'officially', 'named', 'PROPNAME', 'PROPNAME', 'going', 'Tennessee', 'Impact', 'obsessed', 'Maple', 'Story', 'P', 'rewrite', 'story', 'thinks', 'accidentally', 'deleted', 'sad', 'drawing', 'PROPNAME', 'PROPNAME', 'going', 'Pennsylvania', 'tomorrow', 'got', 'hair', 'cut', 'wants', 'Sims', '3', 'xD', 'rearranged', 'room', 'Drastically', 'Its', 'different', 'D', 'boredd', 'play', 'Decode', 'piano', 'D', 'aerobics', 'Family', 'Force', '5', 'Epic', 'goin', 'shopping', 'tomorrow', 'mah', 'friends', 'D', 'writing', 'essays', 'seventeen', 'P', '17', 'days', 'kittens', 'The', 'girl', 'PROPNAME', 'I', 'need', 'help', 'boy', 'PROPNAME', 'PROPNAME', 'PROPNAME', 'PROPNAME', 'VOTE', 'beat', 'sister', 'arm', 'wrestle', 'D', 'sad', 'thinking', 'kitties', 'home', 'bed', 'xD', 'poor', 'kitties', 'drew', 'bluebird', 'drawing', 'seagull', 'Pennsylvania', 'P', 'listening', 'Muse', 'D', 'cut', 'hair', 'Short', 'Again', 'grr', 'school', 'sucks', 'played', 'DD', 'Muahaha', 'What', 'actress', 'I', 'look', 'like', 'o', 'I', 'longer', 'tell', 'days', 'nights', 'The', 'moon', 'glows', 'eerie', 'red', 'I', 'swear', 'covered', 'blood', 'What', 'What', 'A', 'people', 'concerned', 'temporary', 'pleasures', 'world', 'salvation', 'I', 'convinced', 'end', 'As', 'I', 'raise', 'head', 'heavens', 'look', 'moon', 'stars', 'begin', 'fall', 'masquerade', 'seventeen', 'days', 'Christmas', 'day', 'Seventeen', 'days', 'Merry', 'Christmas', 'D', 'feels', 'sickly', 'hahaha', 'school', 'hates', 'writers', 'block', 'rearranging', 'room', 'D', 'Much', 'fun', 'Mystery', 'Retreat', 'blasting', 'screamo', 'theres', 'hard', 'time', 'D', 'bored', 'PSATs', 'got', 'charcoals', 'Christmas', 'D', 'cookies', 'soldiers', 'D', 'sitting', 'bed', 'cats', 'Yes', 'I', 'awesome', '73', 'pages15000', 'words', 'continuous', 'narrative', 'Almost', 'ready', 'editing', 'phase', 'PROPNAME', 'PROPNAME', 'Now', 'getting', 'complicated', 'stuff', 'Fun', '17', 'days', 'I', 'missed', 'yesterdays', 'count', 'packed', 'xD', 'German', 'riddles', 'Fun', 'fun', 'P', 'PROPNAMEs', 'swing', 'dancing', 'equals', 'epic', 'xD', 'Can', 'You', 'See', 'My', 'Eyes', 'Are', 'Shining', 'Bright', 'Cause', 'Im', 'Out', 'Here', 'On', 'The', 'Other', 'Side', 'Of', 'A', 'Jet', 'Black', 'Hotel', 'Mirror', 'going', 'HOPE', 'game', 'night']", "['1.35', '4.75', '2.85', '4.55', '4.4']", Post_0

        score = personality_model(tokens)

        if len(tokens) < 5:
            print_tui([4, 4, 4, 4, 4])

        else:
            print_tui(score)

        send_recognition(name, score)


        # send_document(name, tokens)
        # print(tokens)


# main()

def model_interface():
    global pub_recognition
    rospy.init_node('KIST_model_interface', anonymous=False)
    rospy.Subscriber("documentResult", String, callback_document)
    pub_recognition = rospy.Publisher("recognitionResult", String, queue_size=100)

    rospy.spin()

if __name__ == '__main__':
    model_interface()
#
# data1 = ['CDC', 'latest', 'report', 'Autism', 'effects', '1', '110', 'children', 'born', 'day', '1', '70', 'boys', 'Put', 'status', '1', 'HOUR', 'know', 'AUTISM', 'Let', 'childrens', 'voices', 'heard', 'Heres', '2010', 'increasing', 'awareness', 'research', 'proactively', 'finding', 'answers', 'going', 'pool', 'later', 'writing', 'Psychology', 'essay', 'ew', 'Chemistry', 'art', 'special', 'going', 'anymore', 'better', 'I', 'guess', 'finishing', 'Psychology', 'essays', 'wants', 'cut', 'hair', 'finishing', 'homework', 'On', 'Algebra', 'II', 'going', 'art', 'class', 'writing', '50', 'page', 'mark', 'wants', 'keyboard', 'German', 'classs', 'sick', 'And', 'bored', 'Someone', 'talk', 'meee', 'starts', 'Psychology', 'tomorrow', 'haha', 'PROPNAME', 'awesome', 'xD', 'Chemistryy', 'soo', 'hardd', 'doin', 'German', 'homework', 'First', 'Mission', 'Decode', 'Paramore', 'goin', 'college', 'wants', 'kitten', 'I', 'shall', 'PROPNAME', 'sitting', 'outside', 'Pennsylvania', 'kittens', 'officially', 'named', 'PROPNAME', 'PROPNAME', 'going', 'Tennessee', 'Impact', 'obsessed', 'Maple', 'Story', 'P', 'rewrite', 'story', 'thinks', 'accidentally', 'deleted', 'sad', 'drawing', 'PROPNAME', 'PROPNAME', 'going', 'Pennsylvania', 'tomorrow', 'got', 'hair', 'cut', 'wants', 'Sims', '3', 'xD', 'rearranged', 'room', 'Drastically', 'Its', 'different', 'D', 'boredd', 'play', 'Decode', 'piano', 'D', 'aerobics', 'Family', 'Force', '5', 'Epic', 'goin', 'shopping', 'tomorrow', 'mah', 'friends', 'D', 'writing', 'essays', 'seventeen', 'P', '17', 'days', 'kittens', 'The', 'girl', 'PROPNAME', 'I', 'need', 'help', 'boy', 'PROPNAME', 'PROPNAME', 'PROPNAME', 'PROPNAME', 'VOTE', 'beat', 'sister', 'arm', 'wrestle', 'D', 'sad', 'thinking', 'kitties', 'home', 'bed', 'xD', 'poor', 'kitties', 'drew', 'bluebird', 'drawing', 'seagull', 'Pennsylvania', 'P', 'listening', 'Muse', 'D', 'cut', 'hair', 'Short', 'Again', 'grr', 'school', 'sucks', 'played', 'DD', 'Muahaha', 'What', 'actress', 'I', 'look', 'like', 'o', 'I', 'longer', 'tell', 'days', 'nights', 'The', 'moon', 'glows', 'eerie', 'red', 'I', 'swear', 'covered', 'blood', 'What', 'What', 'A', 'people', 'concerned', 'temporary', 'pleasures', 'world', 'salvation', 'I', 'convinced', 'end', 'As', 'I', 'raise', 'head', 'heavens', 'look', 'moon', 'stars', 'begin', 'fall', 'masquerade', 'seventeen', 'days', 'Christmas', 'day', 'Seventeen', 'days', 'Merry', 'Christmas', 'D', 'feels', 'sickly', 'hahaha', 'school', 'hates', 'writers', 'block', 'rearranging', 'room', 'D', 'Much', 'fun', 'Mystery', 'Retreat', 'blasting', 'screamo', 'theres', 'hard', 'time', 'D', 'bored', 'PSATs', 'got', 'charcoals', 'Christmas', 'D', 'cookies', 'soldiers', 'D', 'sitting', 'bed', 'cats', 'Yes', 'I', 'awesome', '73', 'pages15000', 'words', 'continuous', 'narrative', 'Almost', 'ready', 'editing', 'phase', 'PROPNAME', 'PROPNAME', 'Now', 'getting', 'complicated', 'stuff', 'Fun', '17', 'days', 'I', 'missed', 'yesterdays', 'count', 'packed', 'xD', 'German', 'riddles', 'Fun', 'fun', 'P', 'PROPNAMEs', 'swing', 'dancing', 'equals', 'epic', 'xD', 'Can', 'You', 'See', 'My', 'Eyes', 'Are', 'Shining', 'Bright', 'Cause', 'Im', 'Out', 'Here', 'On', 'The', 'Other', 'Side', 'Of', 'A', 'Jet', 'Black', 'Hotel', 'Mirror', 'going', 'HOPE', 'game', 'night']
# data2 = ['tryin', 'figure', 'w', 'plane', 'ticket', 'NYC', '24th', 'w', 'hang', 'FML', 'lets', 'drunk', 'somebody', 'livin', 'dream', 'Findin', 'steady', 'pay', 'check', 'year', 'Lovin', 'Hawaiian', 'pizza', 'I', 'wish', 'I', 'kiss', 'Because', 'I', 'know', 'taste', 'betrayal', 'WHITEOUT', 'credit', 'card', 'roulette', 'cost', '8110FML', 'Im', 'better', 'ya', 'ex', 'gon', 'na', 'better', 'ya', 'PSU', 'loss', 'Steelers', 'lossmy', 'life', 'shambles', 'IM', 'football', 'Southsidelife', 'good', 'insert', 'depressing', 'emo', 'lyrics', 'PSU', 'weekend', 'obvi', 'dont', 'think', 'I', 'dont', 'think', 'dont', 'think', 'I', 'dont', 'regrets', 'Maybe', 'break', 'better', 'Football', 'daythats', 'I', 'love', 'Sundays', 'Lovin', 'life', 'shortys', 'like', 'melody', 'head', 'Getting', 'needed', 'TLC', 'momand', 'trap', 'cleaner', 'nice', 'INDEPENDENT', 'know', 'means', '801', 'You', 'dropped', 'wrong', 'girls', 'house', '1801', 'Theres', 'thing', 'wrong', 'girl', 'happen', '801', 'alright', 'morning', 'CMU', 'footballis', 'better', 'ugh', 'rough', 'weekend', 'dying', 'dysentry', 'WPT', 'yes', 'terrible', 'night', 'Southside', 'tonightany', 'takers', 'boot', 'ass', 'American', 'way', 'I', 'got', 'feelin', 'tonights', 'gon', 'na', 'good', 'night', 'This', 'man', 'cave', 'theres', 'women', 'allowed', 'I', 'got', 'jerkoff', 'station', 'Gods', 'sake', 'Deceiving', 'Thats', 'world', 'calls', 'romance', 'Taking', 'frustrations', '300', 'plus', 'pounds', 'gravity', 'motivated', 'iron', 'I', 'know', 'got', 'reason', 'past', 'away', 'Tattoo', 'lower', 'backmight', 'bullseye', 'Just', 'trying', 'play', '1', 'round', 'golf', 'summer', 'thought', 'chick', 'hitting', 'night', 'turned', 'getting', 'attention', 'fun', '20', 'wearing', 'Abercrombie', 'FML', 'job', 'sparkling', 'wiggles', 'I', 'love', 'PSU', 'day', 'work', 'fin', 'right', 'doggie', 'Round', '2', 'Ive', 'better', 'living', 'hazelooking', 'meaning', 'life', 'finding', 'darkness', 'trying', 'justify', 'CMU', 'education', 'exchange', 'PSU', 'atmospherenot', 'working', 'cooler', 'online', 'trust', 'lies', 'core', 'love', ';', 'true', 'love', 'trust', 'working', 'week', 'time', 'play', 'gon', 'na', 'little', 'bit', 'sideways', 'wicked', 'salacity', 'leads', 'anger', 'frustration', 'youre', 'sure', 'love', 'youre', 'sure', 'let', 'taking', 'step', 'forward', '2', 'steps', 'backstory', 'life', 'lonely', 'night', 'hold', 'memories', 'better', 'days', 'exhausted', 'recovery', 'day', 'ugh', 'rough', 'shape', 'Dont', 'worry', 'people', 'past', 'theres', 'reason', 'didnt', 'future', 'said', 'probably', 'dont', 'remember', 'I', 'said', 'memory', 'like', 'burning', 'end', 'midnight', 'cigarette', 'Trying', 'learn', 'build', 'cowboy', 'hat', 'Natty', 'light', 'boxes', 'noddin', 'head', 'like', 'Yeah', 'movin', 'hips', 'like', 'Yeah', 'Just', 'got', 'pizza', 'delivered', 'house', 'Pittsburgh', 'chick', 'I', 'went', 'high', 'school', 'Coudersportfeel', 'better', 'life', 'right', 'I', 'dont', 'feel', 'like', 'watching', 'football', 'I', 'watch', 'good', 'moviePROPNAMEmanhood', 'question', 'Jammin', 'PROPNAME', 'new', 'Trapp', 'piano', 'All', 'classy', 'broads', 'hit', 'half', 'price', 'wine', 'night', 'right', 'anotha', 'day', 'anotha', 'dolla', 'hang', 'tight', 'cause', 'gon', 'na', 'wilder', '8', 'second', 'ride', 'ridiculous', 'FML', 'single', 'youre', 'willing', 'try', 'hard', 'PHI', 'callin', 'Ready', 'long', 'weekend', 'false', 'alarm', 'invincible', 'State', 'blows', 'opportunityshouldnt', 'expected', 'I', 'officially', 'dumbest', 'person', 'entire', 'worldmy', 'life', 'sucks', 'Chitown', 'weekend', 'False', 'alarmstill', 'invincible', 'Is', 'May', 'Turkey', 'Bowl', 'Beaver', 'Stadiumsick', 'guess', 'thats', 'youre', 'Tailgating', 'Wendys', 'parking', 'lot', 'new', 'thing', 'Composing', 'script', 'greek', 'singwill', 'definitely', 'epic', 'LMFAO', 'tryin', 'streak', 'alive', 'To', 'feel', 'pain', 'feel', 'Go', 'inside', 'eachothers', 'minds', 'wed', 'find', 'Look', 'shit', 'eachothers', 'eyes', 'Man', 'date', 'HokkaidoSMAAART', 'scars', 'heal', 'glory', 'fades', 'left', 'memories', 'pain', 'hurts', 'minute', 'life', 'short', 'live', 'cause', 'chicks', 'dig', 'needs', 'formal', 'date', 'Friday', 'nightstat', 'Cleveland', 'celebrating', 'birthdays', 'King', 'New', 'Years', 'burg', 'cruisin', 'Caribbeangonna', 'nice', 'little', 'weekend', 'Im', 'pretty', 'sure', 'theres', 'lot', 'life', 'ridiculously', 'good', 'looking', 'And', 'I', 'plan', 'finding', '85', 'sunny', 'Jamaica', '82', 'sunny', 'Grand', 'CaymanSMICK', 'Puzzling', 'face', 'Women', 'lie', 'men', 'lie', 'numbers', 'dont', 'lie', '4', 'rounds', 'credit', 'card', 'roulette', 'shots', 'paylove', 'Trapp', 'mandatedinner', 'movie', 'better', 'Lately', 'Ive', 'hard', 'reach', 'Ive', 'long', 'Everybody', 'private', 'world', 'Are', 'calling', 'Are', 'trying', 'Are', 'reaching', 'Im', 'reaching']
# data3 = ['Desperately', 'seeking', 'room', 'slash', 'room', 'mate', 'Can', 'help', 'leads', 'Attention', 'If', 'misfortune', 'reading', 'I', 'regret', 'inform', 'Ive', 'seriously', 'injured', 'car', 'accident', 'Okay', 'maybe', 'I', 'lose', 'game', 'Love', 'Dead', 'Anyone', 'think', 'help', 'escorting', 'airport', 'Friday', 'morning', 'I', 'pay', 'undying', 'gratitude', 'awkward', 'silences', 'Plan', 'A', 'officially', 'scrapped', 'Good', 'thing', 'I', 'plan', 'B', 'right', 'Airportage', '545', 'So', 'like', 'bored', 'Saturday', 'night', 'Failingthe', 'best', 'thing', 'succeeding', 'Sick', 'Im', 'going', 'blame', 'Northernerswhy', 'Why', 'Back', 'vacation', 'Cruise', 'fun', 'I', 'feel', 'like', 'I', 'need', 'vacation', 'previous', 'vacation', 'In', 'Toronto', 'Im', 'getting', 'Canadian', 'Would', 'like', 'failsauce', 'epic', 'fail', 'Back', '813the', 'place', 'So', 'Zombieland', 'Great', 'movieor', 'greatest', 'movie', 'If', 'tree', 'falls', 'woods', 'hear', 'lose', 'game', 'So', 'theres', 'SuperBowl', 'On', 'right']
#
# print(personality_model(data3))
# print(personality_model(data2))