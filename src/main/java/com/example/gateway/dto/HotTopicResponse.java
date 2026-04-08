package com.example.gateway.dto;

import java.util.List;

public record HotTopicResponse(String platform, String date, List<HotTopicItem> topics, String note) {
}
