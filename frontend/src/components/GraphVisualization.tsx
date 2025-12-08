import { useEffect, useState, useRef } from 'react'
import ForceGraph2D, { ForceGraphMethods, NodeObject } from 'react-force-graph-2d'
import { apiClient } from '@/services/api'

interface GraphNode {
  id: string
  label: string
  properties: Record<string, unknown>
  x?: number
  y?: number
  __bckgDimensions?: [number, number]
}

interface GraphLink {
  source: string
  target: string
  type: string
  properties: Record<string, unknown>
}

interface GraphData {
  nodes: GraphNode[]
  links: GraphLink[]
}

const GraphVisualization = () => {
  const [data, setData] = useState<GraphData>({ nodes: [], links: [] })
  const [loading, setLoading] = useState(true)
  const fgRef = useRef<ForceGraphMethods | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await apiClient.get('/graph/data')
        if (response.data?.success && response.data?.data) {
          setData(response.data.data as GraphData)
        }
      } catch (error) {
        console.error('Error fetching graph data:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [])

  const handleNodeClick = (node: NodeObject<GraphNode>) => {
    if (fgRef.current) {
      fgRef.current.centerAt(node.x ?? 0, node.y ?? 0, 1000)
      fgRef.current.zoom(8, 2000)
    }
  }

  if (loading) {
    return <div className="flex justify-center items-center h-64 text-gray-400">Loading Knowledge Graph...</div>
  }

  return (
    <div className="w-full h-[600px] border border-gray-700 rounded-lg overflow-hidden bg-gray-900">
      <ForceGraph2D
        ref={fgRef}
        graphData={data}
        nodeLabel="label"
        nodeAutoColorBy="label"
        linkDirectionalArrowLength={3.5}
        linkDirectionalArrowRelPos={1}
        onNodeClick={handleNodeClick}
        backgroundColor="#111827"
        nodeCanvasObject={(node: NodeObject<GraphNode>, ctx, globalScale) => {
          const label = (node.properties?.name as string) || node.label || node.id
          const fontSize = 12 / globalScale
          ctx.font = `${fontSize}px Sans-Serif`
          const textWidth = ctx.measureText(label).width
          const textPadding = fontSize * 0.2
          const bckgDimensions: [number, number] = [
            textWidth + textPadding,
            fontSize + textPadding,
          ]

          ctx.fillStyle = 'rgba(255, 255, 255, 0.2)'
          ctx.fillRect(
            (node.x ?? 0) - bckgDimensions[0] / 2,
            (node.y ?? 0) - bckgDimensions[1] / 2,
            bckgDimensions[0],
            bckgDimensions[1],
          )

          ctx.textAlign = 'center'
          ctx.textBaseline = 'middle'
          ctx.fillStyle = '#60A5FA'
          ctx.fillText(label, node.x ?? 0, node.y ?? 0)

          node.__bckgDimensions = bckgDimensions
        }}
        nodePointerAreaPaint={(node: NodeObject<GraphNode>, color, ctx) => {
          ctx.fillStyle = color
          const bckgDimensions = node.__bckgDimensions
          if (bckgDimensions) {
            ctx.fillRect(
              (node.x ?? 0) - bckgDimensions[0] / 2,
              (node.y ?? 0) - bckgDimensions[1] / 2,
              bckgDimensions[0],
              bckgDimensions[1],
            )
          }
        }}
      />
    </div>
  )
}

export default GraphVisualization
