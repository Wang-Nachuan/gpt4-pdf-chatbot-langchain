import { OpenAI } from 'langchain/llms/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';

const CONDENSE_PROMPT = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;

const QA_PROMPT = `Please use the following pieces of context to answer the question at the end.
Please always follow these rules while answering questions:
1. If you don't know the answer, just say you don't know. DO NOT try to make up an answer.
2. If the question is not related to the context, first directly point it out, then try to find the most related content based on your knowledge.
3. I will referr a paper in the format of "[pulication year], [paper title]".
4. I will specify the paper I want to discuss in the format of "Now discuss: [pulication year], [paper title]", by default all following questions are about this paper, unless I explicitly mentioned another paper in the same format. Please tell me whether you know this paper when I say "Now discuss: [pulication year], [paper title]".
5. You answer should mainly focus on the content of paper we are discussing.

{context}

Question: {question}
Helpful answer in markdown:`;

export const makeChain = (vectorstore: PineconeStore) => {
  const model = new OpenAI({
    temperature: 0, // increase temepreature to get more creative answers
    // modelName: 'gpt-3.5-turbo', //change this to gpt-4 if you have access
    // modelName: 'gpt-4,
    modelName: 'gpt-4',
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(),
    {
      qaTemplate: QA_PROMPT,
      questionGeneratorTemplate: CONDENSE_PROMPT,
      returnSourceDocuments: true, //The number of source documents returned is 4 by default
    },
  );
  return chain;
};
