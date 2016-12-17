"""
sumarize.py
"""
import pickle

def main():
    pkl_dump_data = open('Classify_Data/summarize.pkl','rb')
    dump_data = pickle.load(pkl_dump_data)
    print("---------SUMMARY STATISTICS----------------")
    print("Number of users: %d" % len(dump_data[0].keys()))
    print("Number of messages: %d" % len(dump_data[1]))
    print("Number of communitites discovered: %d" % len(dump_data[2].keys()))
    print("Average number of users per community: %d" % len(dump_data[2][0]))
    print("Number of instances per class: %d positive and %d negative" % (dump_data[3], dump_data[4]))
    print("Example of positive class: %s" % dump_data[5])
    print("Example of negative class: %s" % dump_data[6])
    
    file = open("Summarize/summary.txt","w")
    
    file.write("Number of users: %d \n" % len(dump_data[0].keys()))
    file.write("Number of messages: %d \n" % len(dump_data[1]))
    file.write("Number of communitites discovered: %d \n" % len(dump_data[2].keys()))
    file.write("Average number of users per community: %d \n" % len(dump_data[2][0]))    
    file.write("Number of instances per class: %d positive and %d negative \n" % (dump_data[3], dump_data[4]))
    file.write("Example of positive class: %s \n" % dump_data[5])
    file.write("Example of negative class: %s \n" % dump_data[6])
    
    file.close()

if __name__ == '__main__':
    main()