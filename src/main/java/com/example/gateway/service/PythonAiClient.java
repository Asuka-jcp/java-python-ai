package com.example.gateway.service;

import com.example.gateway.dto.*;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

@Service
public class PythonAiClient {

    private final RestTemplate restTemplate;
    private final String pythonBaseUrl;

    public PythonAiClient(RestTemplate restTemplate,
                          @Value("${python.ai.base-url:http://localhost:8000}") String pythonBaseUrl) {
        this.restTemplate = restTemplate;
        this.pythonBaseUrl = pythonBaseUrl;
    }

    public HotTopicResponse fetchHotTopics(String platform) {
        HttpEntity<HotTopicRequest> entity = jsonEntity(new HotTopicRequest(platform));
        ResponseEntity<HotTopicResponse> response = restTemplate.postForEntity(
                pythonBaseUrl + "/api/ai/hot-topics", entity, HotTopicResponse.class
        );
        return response.getBody();
    }

    public RewriteResponse rewrite(String text) {
        HttpEntity<RewriteRequest> entity = jsonEntity(new RewriteRequest(text));
        ResponseEntity<RewriteResponse> response = restTemplate.postForEntity(
                pythonBaseUrl + "/api/ai/rewrite", entity, RewriteResponse.class
        );
        return response.getBody();
    }

    public ChatResponse chat(String message) {
        HttpEntity<ChatRequest> entity = jsonEntity(new ChatRequest(message));
        ResponseEntity<ChatResponse> response = restTemplate.postForEntity(
                pythonBaseUrl + "/api/ai/chat", entity, ChatResponse.class
        );
        return response.getBody();
    }

    private <T> HttpEntity<T> jsonEntity(T body) {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        return new HttpEntity<>(body, headers);
    }
}
