package com.example.gateway.dto;

public record ChatResponse(
        String intent,
        String answer,
        String note
) {
}
