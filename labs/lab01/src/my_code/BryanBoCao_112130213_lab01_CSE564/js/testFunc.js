<body>
<script>
                    var data = [4, 1, 6, 2, 8, 9];
                    var body = d3.select("body")
                                .selectAll("span")
                                .data(data)
                                .enter()
                                .append("span")
                                .text(function(d) { return d + " "; });
</script>
</body>
