'use client'
import React, { useState, ChangeEvent, useEffect, useRef } from 'react';
import { Button } from "@/components/ui/button";
import { LuSend} from "react-icons/lu";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import Loader from '@/components/shared/Loader';
import HomeStarter from '@/components/shared/HomeStarter';
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { 
  Accordion, 
  AccordionContent, 
  AccordionItem, 
  AccordionTrigger 
} from "@/components/ui/accordion";
import { LuExternalLink } from "react-icons/lu"; // Icon for links

interface SourceDocumentMetadata {
  file_path?: string;
  doc_name?: string;
  source?: string;
  date?: string;
  num_pages?: number;
  _id?: string;
  _collection_name?: string;
  score?: number;
  [key: string]: any; // Allow extra fields
}

interface SourceDocument {
  page_content: string;
  metadata: SourceDocumentMetadata;
}

interface Message {
  type: 'user' | 'bot';
  text: string;
  source_documents?: SourceDocument[]; // Add source documents
  audio_base64?: string;
  suggested_video?: {
    title: string;
    link: string;
    thumbnail: string;
  };
}

export default function Home() {
  const [userQuery, setUserQuery] = useState<string>("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [useReranker, setUseReranker] = useState<boolean>(false); 
  const chatContainerRef = useRef<HTMLDivElement>(null);

  const handleInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    setUserQuery(e.target.value);
  };

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  const sendMessage = async (message: string) => {
    setIsLoading(true);
    if (message.trim() === "") {
      setIsLoading(false);
      return;
    }

    setMessages(prevMessages => [...prevMessages, { type: 'user', text: message }]);
    setUserQuery(""); 

    const endpoint = useReranker 
      ? 'http://localhost:8000/query/hybrid-rerank/' 
      : 'http://localhost:8000/query/hybrid/';
    
    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: message,
        }),
      });

      if (response.ok) {
        // Update expected response structure
        const data: { answer: string, source_documents: SourceDocument[] } = await response.json();
        console.log('Response from backend', data);

        setMessages(prevMessages => [...prevMessages, { 
          type: 'bot', 
          text: data.answer, 
          source_documents: data.source_documents // Store source documents
        }]);
      } else {
        console.error('Error in response:', response.statusText);
        // Add a bot message indicating an error
        setMessages(prevMessages => [...prevMessages, { 
          type: 'bot', 
          text: `Sorry, I encountered an error: ${response.statusText}. Please try again.`
        }]);
      }
    } catch (error) {
      console.error('Error sending message:', error);
      // Add a bot message indicating a network or other error
      setMessages(prevMessages => [...prevMessages, { 
        type: 'bot', 
        text: `Sorry, I couldn't connect to the backend. Please check if it's running.`
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  // Helper to get a display name for a document
  const getDocDisplayName = (metadata: SourceDocumentMetadata): string => {
    return metadata.doc_name || metadata.file_path?.split('\\').pop()?.split('/').pop() || metadata.source || 'Source Document';
  }

  // Helper to get a source link (if available)
  const getSourceLink = (metadata: SourceDocumentMetadata): string | null => {
    // Prioritize 'source' if it looks like a URL, otherwise 'file_path'
    if (metadata.source && (metadata.source.startsWith('http://') || metadata.source.startsWith('https://'))) {
      return metadata.source;
    }
    // You might not be able to link directly to local file_paths from the browser
    // For now, return null if it's not a URL
    return null; 
    // Alternatively, display the file_path as text: return metadata.file_path || null;
  }

  return (
    <>
      <main className="flex min-h-screen flex-col items-center px-4 sm:px-8 md:px-14 py-24 pb-32"> 
        
        {/* Mode Toggle Switch */} 
        {/* <div className="fixed top-4 right-4 z-50 bg-white p-3 rounded-lg shadow-md border border-gray-200 flex items-center space-x-2">
          <Switch 
            id="reranker-toggle"
            checked={useReranker}
            onCheckedChange={setUseReranker}
            aria-label="Toggle Reranker Mode"
          />
          <Label htmlFor="reranker-toggle" className="text-sm font-medium">
            Use Reranker ({useReranker ? "On" : "Off"})
          </Label>
        </div> */}

        <section ref={chatContainerRef} className="w-full md:w-3/4 lg:w-2/3 h-full flex flex-col gap-3 overflow-y-auto scroll-smooth"> 
          {messages.length === 0 && 
            <HomeStarter />
          }
          {messages.map((message, index) => (
            <div key={index} className="flex flex-row gap-3 my-2 z-40">
              <Avatar className='z-20'>
                <AvatarImage src={message.type === 'user' ? "./useres.png" : "./user2.png"} />
                <AvatarFallback>{message.type === 'user' ? "CN" : "BOT"}</AvatarFallback>
              </Avatar>
              <div className='text-xs md:text-sm lg:text-base flex-1 break-words w-full'> 
                <Markdown remarkPlugins={[remarkGfm]} className="prose prose-sm md:prose-base max-w-none">{message.text}</Markdown>
                
                {/* Display Source Documents for Bot messages */} 
                {message.type === 'bot' && message.source_documents && message.source_documents.length > 0 && (
                  <div className="mt-4">
                    <h4 className="text-xs font-semibold text-gray-600 mb-2">References:</h4>
                    <Accordion type="single" collapsible className="w-full">
                      {message.source_documents.map((doc, docIndex) => {
                        const displayName = getDocDisplayName(doc.metadata);
                        const sourceLink = getSourceLink(doc.metadata);
                        return (
                          <AccordionItem key={docIndex} value={`item-${docIndex}`} className="border-b border-orange-200">
                            <AccordionTrigger className="text-xs md:text-sm text-left hover:no-underline py-2 px-3 bg-orange-50 hover:bg-orange-100 rounded-md">
                              {displayName}
                              {doc.metadata.score && (
                                <span className="ml-2 text-xs text-gray-500 font-normal">(Score: {doc.metadata.score.toFixed(3)})</span>
                              )}
                            </AccordionTrigger>
                            <AccordionContent className="text-xs md:text-sm pt-2 pb-3 px-3 bg-white">
                              <p className="mb-2 whitespace-pre-wrap">{doc.page_content}</p>
                              {sourceLink ? (
                                <a 
                                  href={sourceLink} 
                                  target="_blank" 
                                  rel="noopener noreferrer"
                                  className="text-orange-600 hover:text-orange-800 text-xs inline-flex items-center gap-1"
                                >
                                  Open Source <LuExternalLink /> 
                                </a>
                              ) : (
                                doc.metadata.file_path && (
                                  <span className="text-gray-500 text-xs">Source: {doc.metadata.file_path}</span>
                                )
                              )}
                            </AccordionContent>
                          </AccordionItem>
                        );
                      })}
                    </Accordion>
                  </div>
                )}
              </div>
            </div>
          ))}

          {isLoading && (
            <Loader />
          )}
        </section>
      </main>

      <footer className="flex justify-center z-40 bg-white mt-3">
        <div className="my-2 p-2 mx-2 w-full md:w-3/4 lg:w-2/3 fixed bottom-0 z-40 bg-white">
          <div className="flex flex-row gap-2 border-[1.5px] border-orange-600 justify-center py-2 px-4 rounded-2xl z-40 bg-white">
            <input
              type="text"
              value={userQuery}
              onChange={handleInputChange}
              className="w-full border-none outline-none z-50 text-xs md:text-sm lg:text-base border-orange-500"
              placeholder="Hi! How can I help you?"
              onKeyPress={(e) => {
                if (e.key === 'Enter') {
                  sendMessage(userQuery);
                }
              }}
            />
            <Button 
              type="submit" 
              className="rounded-xl bg-orange-400 hover:bg-orange-600 font-semibold transition ease-in-out disabled:opacity-50" 
              onClick={() => sendMessage(userQuery)}
              disabled={isLoading || userQuery.trim() === ""} // Disable button when loading or input is empty
            >
              <LuSend className="text-lg text-white font-semibold" />
            </Button>
          </div>
          <p className='text-[11px] items-center text-center mt-1'>Disclaimer: The responses are AI-Generated. It may contain mistakes.</p>
        </div>
      </footer>
    </>
  );
}