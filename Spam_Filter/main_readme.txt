Welcome to the CSDMC2010 SPAM corpus, which is one of the datasets for 
the data mining competition associated with ICONIP 2010.

This dataset is composed of a selection of mail messages, suitable for 
use in testing spam filtering systems.  

------------------------------------------------------
Pertinent points

  - All headers are reproduced in full.  Some address obfuscation has taken
    place, and hostnames in some cases have been replaced with
    "csmining.org" (which has a valid MX record) and with most of the recipents
    replaced with 'hibody.csming.org' In most cases
    though, the headers appear as they were received.

  - All of these messages were posted to public fora, were sent to me in the
    knowledge that they may be made public, were sent by me, or originated as
    newsletters from public mail lists. A part of the data is from other 
    public corpus(es), however, for some reason, information will be open
    after the competion.

  - Copyright for the text in the messages remains with the original senders.

------------------------------------------------------
The corpus file -- CSDMC2010_SPAM.tar.bz2 

On Linux platforms, it can be extracted by command 
tar -xjf CSDMC2010_SPAM.tar.bz2 -C email/

In an MS Windows environment, use the bzip2 software
http://gnuwin32.sourceforge.net/packages/bzip2.htm


------------------------------------------------------
The corpus description
The dataset contains two parts:

  - TRAINING: 2999 messages with a mixture of non-spam messages (HAM) spam messages (SPAM), all received from non-spam-trap sources.
  - TESTING: 1328 messages with a mixture of non-spam messages (HAM) spam messages (SPAM), all received from non-spam-trap sources.  	
  	SPAM.label contains the labels of the emails, with 1 stands for a 
  	HAM and 0 stands for a SPAM.
 

  
------------------------------------------------------
The email format description
 
The format of the .eml file is definde in RFC822, and information on recent 
standard of email, i.e., MIME (Multipurpose Internet Mail Extensions) can be
find in RFC2045-2049.
  
Please direct any questions regarding this dataset to <bantao>at<nict>dot<go>dot<jp>.

