original = r'D:\\Projects\\Python\\8980\\flametest\\flame_model\\flame_static_embedding.pkl'
destination = r'D:\\Projects\\Python\\8980\\flametest\\flame_model\\flame_static_embedding_new.pkl'

content = ''
outsize = 0
with open(original, 'rb') as infile:
    
    content = infile.read()
with open(destination, 'wb') as output:
    
    for line in content.splitlines():
        outsize += len(line) + 1
        output.write(line + str.encode('\n'))

print("Done. Saved %s bytes." % (len(content)-outsize))