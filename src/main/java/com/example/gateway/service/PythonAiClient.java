package com.example.gateway.service;

import com.example.gateway.dto.HotTopicRequest;
import com.example.gateway.dto.HotTopicResponse;
import com.example.gateway.dto.RewriteRequest;
import com.example.gateway.dto.RewriteResponse;
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
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<HotTopicRequest> entity = new HttpEntity<>(new HotTopicRequest(platform), headers);
        ResponseEntity<HotTopicResponse> response = restTemplate.postForEntity(
                pythonBaseUrl + "/api/ai/hot-topics", entity, HotTopicResponse.class
        );
        return response.getBody();
    }

    public RewriteResponse rewrite(String text) {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<RewriteRequest> entity = new HttpEntity<>(new RewriteRequest(text), headers);
        ResponseEntity<RewriteResponse> response = restTemplate.postForEntity(
                pythonBaseUrl + "/api/ai/rewrite", entity, RewriteResponse.class
        );
        return response.getBody();
    }
}
