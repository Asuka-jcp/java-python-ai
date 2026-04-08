package com.example.gateway.controller;

import com.example.gateway.dto.HotTopicResponse;
import com.example.gateway.dto.RewriteRequest;
import com.example.gateway.dto.RewriteResponse;
import com.example.gateway.service.PythonAiClient;
import jakarta.validation.Valid;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;

import java.util.Set;

@RestController
@RequestMapping("/api/gateway")
public class GatewayController {

    private static final Set<String> SUPPORTED = Set.of("微博", "知乎", "抖音", "小红书", "B站", "X", "Reddit");
    private final PythonAiClient pythonAiClient;

    public GatewayController(PythonAiClient pythonAiClient) {
        this.pythonAiClient = pythonAiClient;
    }

    @GetMapping("/platforms")
    public Set<String> platforms() {
        return SUPPORTED;
    }

    @GetMapping("/hot-topics")
    public HotTopicResponse hotTopics(@RequestParam String platform) {
        if (!SUPPORTED.contains(platform)) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "不支持的平台: " + platform);
        }
        HotTopicResponse response = pythonAiClient.fetchHotTopics(platform);
        if (response == null) {
            throw new ResponseStatusException(HttpStatus.BAD_GATEWAY, "Python 服务无响应");
        }
        return response;
    }

    @PostMapping("/rewrite")
    public RewriteResponse rewrite(@Valid @RequestBody RewriteRequest request) {
        RewriteResponse response = pythonAiClient.rewrite(request.text());
        if (response == null) {
            throw new ResponseStatusException(HttpStatus.BAD_GATEWAY, "Python 服务无响应");
        }
        return response;
    }
}
