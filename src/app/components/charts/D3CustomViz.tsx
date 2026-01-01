import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface D3CustomVizProps {
  data?: {
    nodes?: Array<{ id: string; group: number; confidence: number }>;
    links?: Array<{ source: string; target: string; value: number }>;
    circularData?: Array<{ feature: string; value: number; angle: number }>;
  };
  type?: 'force-directed' | 'circular-heatmap' | 'confidence-gauge';
  title?: string;
  width?: number;
  height?: number;
}

const D3CustomViz: React.FC<D3CustomVizProps> = ({
  data,
  type = 'force-directed',
  title = "Custom Visualization",
  width = 600,
  height = 400
}) => {
  const svgRef = useRef<SVGSVGElement>(null);

  // Default data for demonstration
  const defaultNodes = [
    { id: "Video Input", group: 1, confidence: 0.9 },
    { id: "Frame Sampler", group: 1, confidence: 0.85 },
    { id: "Face Detector", group: 2, confidence: 0.92 },
    { id: "BG Model", group: 3, confidence: 0.78 },
    { id: "AV Model", group: 3, confidence: 0.83 },
    { id: "CM Model", group: 3, confidence: 0.88 },
    { id: "RR Model", group: 3, confidence: 0.91 },
    { id: "LL Model", group: 3, confidence: 0.76 },
    { id: "TM Model", group: 3, confidence: 0.89 },
    { id: "Aggregator", group: 4, confidence: 0.87 },
    { id: "Final Result", group: 5, confidence: 0.84 }
  ];

  const defaultLinks = [
    { source: "Video Input", target: "Frame Sampler", value: 1 },
    { source: "Frame Sampler", target: "Face Detector", value: 1 },
    { source: "Face Detector", target: "BG Model", value: 0.8 },
    { source: "Face Detector", target: "AV Model", value: 0.9 },
    { source: "Face Detector", target: "CM Model", value: 0.85 },
    { source: "Face Detector", target: "RR Model", value: 0.92 },
    { source: "Face Detector", target: "LL Model", value: 0.75 },
    { source: "Face Detector", target: "TM Model", value: 0.88 },
    { source: "BG Model", target: "Aggregator", value: 0.78 },
    { source: "AV Model", target: "Aggregator", value: 0.83 },
    { source: "CM Model", target: "Aggregator", value: 0.88 },
    { source: "RR Model", target: "Aggregator", value: 0.91 },
    { source: "LL Model", target: "Aggregator", value: 0.76 },
    { source: "TM Model", target: "Aggregator", value: 0.89 },
    { source: "Aggregator", target: "Final Result", value: 0.87 }
  ];

  const defaultCircularData = [
    { feature: "Compression", value: 0.85, angle: 0 },
    { feature: "Lighting", value: 0.72, angle: 60 },
    { feature: "Temporal", value: 0.91, angle: 120 },
    { feature: "Artifacts", value: 0.68, angle: 180 },
    { feature: "Quality", value: 0.79, angle: 240 },
    { feature: "Audio", value: 0.83, angle: 300 }
  ];

  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove(); // Clear previous render

    // Add title
    svg.append("text")
      .attr("x", width / 2)
      .attr("y", 20)
      .attr("text-anchor", "middle")
      .style("font-size", "16px")
      .style("font-weight", "bold")
      .style("fill", "#374151")
      .text(title);

    if (type === 'force-directed') {
      createForceDirectedGraph(svg);
    } else if (type === 'circular-heatmap') {
      createCircularHeatmap(svg);
    } else if (type === 'confidence-gauge') {
      createConfidenceGauge(svg);
    }
  }, [data, type, title, width, height]);

  const createForceDirectedGraph = (svg: d3.Selection<SVGSVGElement, unknown, null, undefined>) => {
    const nodes = data?.nodes || defaultNodes;
    const links = data?.links || defaultLinks;

    const color = d3.scaleOrdinal(d3.schemeCategory10);

    const simulation = d3.forceSimulation(nodes as any)
      .force("link", d3.forceLink(links).id((d: any) => d.id).distance(80))
      .force("charge", d3.forceManyBody().strength(-300))
      .force("center", d3.forceCenter(width / 2, height / 2));

    // Add links
    const link = svg.append("g")
      .selectAll("line")
      .data(links)
      .enter().append("line")
      .attr("stroke", "#999")
      .attr("stroke-opacity", 0.6)
      .attr("stroke-width", (d: any) => Math.sqrt(d.value) * 3);

    // Add nodes
    const node = svg.append("g")
      .selectAll("circle")
      .data(nodes)
      .enter().append("circle")
      .attr("r", (d: any) => 5 + d.confidence * 15)
      .attr("fill", (d: any) => color(d.group.toString()))
      .attr("stroke", "#fff")
      .attr("stroke-width", 2)
      .call(d3.drag<SVGCircleElement, any>()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended) as any);

    // Add labels
    const labels = svg.append("g")
      .selectAll("text")
      .data(nodes)
      .enter().append("text")
      .text((d: any) => d.id)
      .style("font-size", "10px")
      .style("fill", "#374151")
      .attr("text-anchor", "middle")
      .attr("dy", 3);

    // Add tooltips
    node.append("title")
      .text((d: any) => `${d.id}\nConfidence: ${(d.confidence * 100).toFixed(1)}%`);

    simulation.on("tick", () => {
      link
        .attr("x1", (d: any) => d.source.x)
        .attr("y1", (d: any) => d.source.y)
        .attr("x2", (d: any) => d.target.x)
        .attr("y2", (d: any) => d.target.y);

      node
        .attr("cx", (d: any) => d.x)
        .attr("cy", (d: any) => d.y);

      labels
        .attr("x", (d: any) => d.x)
        .attr("y", (d: any) => d.y);
    });

    function dragstarted(event: any, d: any) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(event: any, d: any) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragended(event: any, d: any) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }
  };

  const createCircularHeatmap = (svg: d3.Selection<SVGSVGElement, unknown, null, undefined>) => {
    const circularData = data?.circularData || defaultCircularData;
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(width, height) / 3;

    const colorScale = d3.scaleSequential(d3.interpolateViridis)
      .domain([0, 1]);

    // Create circular segments
    const arc = d3.arc<any>()
      .innerRadius(radius * 0.3)
      .outerRadius((d: any) => radius * 0.3 + (d.value * radius * 0.6))
      .startAngle((d: any) => (d.angle - 30) * Math.PI / 180)
      .endAngle((d: any) => (d.angle + 30) * Math.PI / 180);

    const g = svg.append("g")
      .attr("transform", `translate(${centerX},${centerY})`);

    // Add segments
    g.selectAll("path")
      .data(circularData)
      .enter().append("path")
      .attr("d", arc)
      .attr("fill", (d: any) => colorScale(d.value))
      .attr("stroke", "#fff")
      .attr("stroke-width", 2)
      .on("mouseover", function(event, d: any) {
        d3.select(this).attr("opacity", 0.8);
        
        // Add tooltip
        const tooltip = svg.append("g")
          .attr("id", "tooltip");
        
        const rect = tooltip.append("rect")
          .attr("x", event.layerX + 10)
          .attr("y", event.layerY - 30)
          .attr("width", 120)
          .attr("height", 40)
          .attr("fill", "rgba(0,0,0,0.8)")
          .attr("rx", 5);
        
        tooltip.append("text")
          .attr("x", event.layerX + 15)
          .attr("y", event.layerY - 15)
          .attr("fill", "white")
          .style("font-size", "12px")
          .text(`${d.feature}: ${(d.value * 100).toFixed(1)}%`);
      })
      .on("mouseout", function() {
        d3.select(this).attr("opacity", 1);
        svg.select("#tooltip").remove();
      });

    // Add labels
    g.selectAll("text")
      .data(circularData)
      .enter().append("text")
      .attr("transform", (d: any) => {
        const angle = d.angle * Math.PI / 180;
        const labelRadius = radius * 1.1;
        const x = Math.cos(angle - Math.PI / 2) * labelRadius;
        const y = Math.sin(angle - Math.PI / 2) * labelRadius;
        return `translate(${x},${y})`;
      })
      .attr("text-anchor", "middle")
      .attr("dy", "0.35em")
      .style("font-size", "11px")
      .style("fill", "#374151")
      .text((d: any) => d.feature);
  };

  const createConfidenceGauge = (svg: d3.Selection<SVGSVGElement, unknown, null, undefined>) => {
    const confidence = 0.84; // Example confidence value
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(width, height) / 3;

    const g = svg.append("g")
      .attr("transform", `translate(${centerX},${centerY})`);

    // Background arc
    const backgroundArc = d3.arc()
      .innerRadius(radius * 0.7)
      .outerRadius(radius)
      .startAngle(-Math.PI / 2)
      .endAngle(Math.PI / 2);

    g.append("path")
      .attr("d", backgroundArc as any)
      .attr("fill", "#E5E7EB");

    // Confidence arc
    const confidenceArc = d3.arc()
      .innerRadius(radius * 0.7)
      .outerRadius(radius)
      .startAngle(-Math.PI / 2)
      .endAngle(-Math.PI / 2 + Math.PI * confidence);

    g.append("path")
      .attr("d", confidenceArc as any)
      .attr("fill", confidence > 0.7 ? "#EF4444" : confidence > 0.5 ? "#F59E0B" : "#10B981");

    // Center text
    g.append("text")
      .attr("text-anchor", "middle")
      .attr("dy", "0.35em")
      .style("font-size", "24px")
      .style("font-weight", "bold")
      .style("fill", "#374151")
      .text(`${(confidence * 100).toFixed(1)}%`);

    g.append("text")
      .attr("text-anchor", "middle")
      .attr("dy", "1.5em")
      .style("font-size", "12px")
      .style("fill", "#6B7280")
      .text("Confidence");

    // Scale markers
    const scaleData = [0, 0.25, 0.5, 0.75, 1];
    scaleData.forEach(value => {
      const angle = -Math.PI / 2 + Math.PI * value;
      const x1 = Math.cos(angle) * radius * 1.05;
      const y1 = Math.sin(angle) * radius * 1.05;
      const x2 = Math.cos(angle) * radius * 1.15;
      const y2 = Math.sin(angle) * radius * 1.15;

      g.append("line")
        .attr("x1", x1)
        .attr("y1", y1)
        .attr("x2", x2)
        .attr("y2", y2)
        .attr("stroke", "#374151")
        .attr("stroke-width", 2);

      g.append("text")
        .attr("x", Math.cos(angle) * radius * 1.25)
        .attr("y", Math.sin(angle) * radius * 1.25)
        .attr("text-anchor", "middle")
        .attr("dy", "0.35em")
        .style("font-size", "10px")
        .style("fill", "#6B7280")
        .text(`${(value * 100).toFixed(0)}%`);
    });
  };

  return (
    <div className="bg-white/50 dark:bg-gray-900/50 backdrop-blur-md border border-gray-200 dark:border-gray-800 rounded-2xl p-6">
      <svg
        ref={svgRef}
        width={width}
        height={height}
        style={{ maxWidth: '100%', height: 'auto' }}
      />
    </div>
  );
};

export default D3CustomViz;