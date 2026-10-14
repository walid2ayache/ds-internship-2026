import csv 

with open('username-password-recovery-code.csv','r', newline='', encoding="utf-8") as file:
    reader=csv.reader(file)

    for row in reader:
        print(row)
 
    file.seek(0)
    dreader=csv.DictReader(file)

    for row in dreader:
        print(row['Username'],row['First name'],row['Last name']) 
    
# with open('username-password-recovery-code.csv','a', newline='', encoding="utf-8") as file:
#     writer=csv.writer(file)
#     writer.writerow(['sdis','233234','3232145','qwd2323','sadi','sadas','qwd23','wqdsd'])
    file.seek(0)
    sample=file.read(1024)
    file.seek(0)
    dialect=csv.Sniffer().sniff(sample,delimiters=',;\t')
    print("Delimiter:        ", repr(dialect.delimiter))
    print("Quote Character:  ", repr(dialect.quotechar))
    print("Double Quote:     ", dialect.doublequote)
    print("Skip Initial Space:", dialect.skipinitialspace)
    has_header = csv.Sniffer().has_header(sample)
    print("Has header:", has_header) 
    # reader=csv.reader(file,dialect)
    # for row in reader:
    #     print(row)
    file.seek(0)
    import pandas as pd
    df=pd.read_csv(file)
    print(df.head())
    print(df.tail())
    print(df.describe())
    print(df.info())
    def mean(n):
        return sum(n)/len(n)
    def calcstd(n):
        m=mean(n)
        variance=sum((x-m)**2 for x in n )/ (len(n)-1) 
        return variance**0.5
    identifiers=df['Identifier'].tolist()
    print("Mean:", mean(identifiers))
    print("Std: ", calcstd(identifiers))
    def count_top5words(text):
        words=text.lower().split()
        wordCount={}
        for word in words:
            wordCount[word]=wordCount.get(word,0)+1
        top5=sorted(wordCount.items(),key=lambda item:item[1],reverse=True)[:5]
        return top5
        
    
    userss=" ".join(df['Username'].astype(str))
    print("top5:" , count_top5words(userss))
 
 